"""ImageWAMV2: minimal subclass of ImageWAM for Arm A.

The only behavioural change is the FLUX.2 text path: instead of the frozen,
no_grad Qwen feature extractor over the full prompt, we run the K-slot generator
(differentiable) and use its K embeddings as the text conditioning.

Wiring notes (why this needs no edits to the base package):
  - The slot generator is attached as a child module of `self.mot`
    (`mot.slot_generator`). The trainer builds the optimizer from
    `model.dit.parameters()` and `self.dit is self.mot`, so the generator's
    parameters (all of Qwen + the K query embeddings) are optimized. The
    trainer's freeze/train toggling on `model.dit` also covers it.
  - `save_checkpoint` serializes `self.mot.state_dict()`, so the generator's
    weights are saved automatically; `load_checkpoint` loads the MoT with
    strict=False, so resuming from a base ImageWAM checkpoint (which has no
    generator keys) just leaves the generator at its freshly loaded Qwen init.
  - `mot.forward` only iterates `self.mixtures`, so attaching an extra child
    module does not affect the MoT computation.
"""
from __future__ import annotations

import torch

from imagewam.models.backbones.imagewam import ImageWAM
from imagewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class ImageWAMV2(ImageWAM):
    def attach_slot_generator(self, slot_generator) -> None:
        # Register under the MoT so the existing trainer/optimizer/checkpoint
        # machinery picks it up unchanged.
        self.mot.add_module("slot_generator", slot_generator)
        logger.info(
            "Attached QwenSlotGenerator (K=%d, out_dim=%d) under mot.slot_generator.",
            slot_generator.K,
            slot_generator.out_dim,
        )

    def _slot_generator(self):
        return getattr(self.mot, "slot_generator", None)

    # NOTE: no @torch.no_grad here. `build_inputs_flux2` (the caller) runs with
    # grad enabled during training, so gradients flow from the image/action loss
    # back into the slot generator (query embeddings + full Qwen).
    def _encode_flux2_text(self, sample):
        cached = sample.get("text_hidden_states")
        if cached is not None:
            # e.g. eval paths that pass precomputed conditioning.
            text_hidden = cached.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            text_mask = sample.get("text_attention_mask")
            if text_mask is None:
                raise ValueError("cached `text_hidden_states` must be paired with `text_attention_mask`.")
            return text_hidden, text_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        slot_generator = self._slot_generator()
        if slot_generator is None:
            # Fallback to the base (frozen, no_grad) encoder.
            return ImageWAM._encode_flux2_text(self, sample)

        prompt = sample.get("instruction", sample.get("prompt", sample.get("task")))
        if prompt is None:
            raise ValueError(
                "Arm A online text path requires an `instruction`/`prompt`/`task` field in the sample."
            )
        video = sample.get("video")
        batch_size = int(video.shape[0]) if isinstance(video, torch.Tensor) and video.ndim == 5 else 1
        prompts = [prompt] * batch_size if isinstance(prompt, str) else list(prompt)

        slots, mask = slot_generator(prompts)
        return slots.to(dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)

    # Inference/eval text path. `infer_action_flux2` calls this; the base routes a
    # prompt through the frozen full-prompt encoder, which would NOT match how Arm A
    # was trained (16 slots). Route the prompt through the slot generator instead so
    # eval uses the same text conditioning as training. A precomputed `context` still
    # falls back to the base behaviour.
    def _prepare_flux2_infer_text(self, prompt, context, context_mask):
        slot_generator = self._slot_generator()
        if slot_generator is not None and prompt is not None and context is None and context_mask is None:
            slots, mask = slot_generator([prompt] if isinstance(prompt, str) else list(prompt))
            return slots.to(dtype=self.torch_dtype), mask.to(device=self.device, dtype=torch.bool)
        return super()._prepare_flux2_infer_text(prompt, context, context_mask)
