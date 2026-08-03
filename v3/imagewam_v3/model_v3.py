"""ImageWAMV3: minimal subclass of ImageWAM for v3 (VL LatentReasoner).

Behavioural change vs. base: the FLUX.2 text conditioning is produced by an
image-grounded Qwen3-VL Coconut reasoner instead of the frozen Qwen3-4B feature
extractor. The t0 frame (the same image the VAE ref-token path uses,
``sample["video"][:, :, 0]``) is fed into the VL model; the reasoner returns the K
reasoning representations (Plan A, ``condition_tokens="reasoning"``) as the whole
text conditioning for the MoT.

Wiring mirrors v2 (no edits to the base package):
  - The reasoner is attached as ``mot.vl_reasoner`` so the trainer's optimizer
    (built from ``model.dit.parameters()``, and ``self.dit is self.mot``) and the
    ``mot.state_dict()`` checkpoint pick it up, and ``mot.train()`` toggles it.
  - ``mot.forward`` only iterates ``self.mixtures``, so the extra child module does
    not affect the MoT computation.
Difference vs. v2: v3 does NOT reuse the base Qwen text encoder (a different model);
the base is built with ``load_text_encoder=False`` and the VL reasoner is loaded
separately in ``runtime_v3``.
"""
from __future__ import annotations

import torch

from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageWAMV3(ImageWAM):
    def attach_reasoner(self, reasoner) -> None:
        self.mot.add_module("vl_reasoner", reasoner)
        logger.info(
            "Attached VLLatentReasoner (mode=%s, K=%d, cond=%s) under mot.vl_reasoner.",
            reasoner.mode,
            reasoner.num_latent,
            reasoner.condition_tokens,
        )

    def _reasoner(self):
        return getattr(self.mot, "vl_reasoner", None)

    @staticmethod
    def _frame_to_pil(frames: torch.Tensor):
        """[B,3,H,W] in [-1,1] RGB -> List[PIL.Image] (uint8 RGB).

        The dataset normalizes frames with mean=std=0.5, so undo it (*0.5+0.5) before
        converting. This is the very frame the VAE ref-token path encodes, just handed
        to the VL processor instead.
        """
        from PIL import Image

        x = frames.float()
        x = (x * 0.5 + 0.5).clamp(0, 1)           # [-1,1] -> [0,1]
        x = (x * 255.0).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 1).cpu().numpy()    # [B, H, W, C]
        return [Image.fromarray(x[i], mode="RGB") for i in range(x.shape[0])]

    @classmethod
    def _t0_to_pil(cls, video: torch.Tensor):
        """sample['video'] [B,C,T,H,W] in [-1,1] -> List[PIL] from t0 = video[:, :, 0]."""
        return cls._frame_to_pil(video[:, :, 0])

    # NOTE: no @torch.no_grad. build_inputs_flux2 runs with grad during training, so
    # gradients flow from the image/action loss into the full-param VL LLM + bot +
    # feedback projector.
    def _encode_flux2_text(self, sample):
        cached = sample.get("text_hidden_states")
        if cached is not None:
            text_hidden = cached.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            text_mask = sample.get("text_attention_mask")
            if text_mask is None:
                raise ValueError("cached `text_hidden_states` must be paired with `text_attention_mask`.")
            return text_hidden, text_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        reasoner = self._reasoner()
        if reasoner is None:
            return ImageWAM._encode_flux2_text(self, sample)

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "v3 online text path requires an `instruction`/`prompt`/`task` field in the sample."
            )
        video = sample.get("video")
        if not (isinstance(video, torch.Tensor) and video.ndim == 5):
            raise ValueError(
                "v3 requires online `sample['video']` [B,C,T,H,W] to derive the t0 frame "
                "for the VL reasoner (no `ref_image_latents` cache path)."
            )
        batch_size = int(video.shape[0])
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)
        images = self._t0_to_pil(video)

        text_hidden, mask = reasoner(prompts, images)
        return text_hidden.to(dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)

    def infer_action_flux2(self, prompt, input_image, *args, **kwargs):
        # Stash the observation frame so the v3 text path can feed it to the reasoner
        # (the base signature of _prepare_flux2_infer_text does not carry the image).
        prev = getattr(self, "_infer_input_image", None)
        self._infer_input_image = input_image
        try:
            return super().infer_action_flux2(prompt, input_image, *args, **kwargs)
        finally:
            self._infer_input_image = prev

    def _prepare_flux2_infer_text(self, prompt, context, context_mask):
        # A precomputed `context` still falls back to the base behaviour.
        if context is not None or context_mask is not None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        reasoner = self._reasoner()
        if reasoner is None:
            return super()._prepare_flux2_infer_text(prompt, context, context_mask)
        img = getattr(self, "_infer_input_image", None)
        if img is None:
            raise ValueError(
                "v3 eval text path needs the observation image; infer_action_flux2 "
                "should have stashed it as self._infer_input_image."
            )
        if img.ndim == 3:
            img = img.unsqueeze(0)                 # [3,H,W] -> [1,3,H,W]
        images = self._frame_to_pil(img)           # List[PIL], len 1 (or B)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        text_hidden, mask = reasoner(prompts, images)
        return text_hidden.to(dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)
