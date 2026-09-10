"""Scheme B of the H3 study: a modality-specific sigma schedule for action tokens.

MiniMax-H3 denoises video and audio in one packed sequence but runs one
flow-matching schedule per modality (sigma shift 12.0 for video vs 3.0 for
audio), conditioning every row on its own noise level. Here the video tokens
keep the stock schedule and the action tokens follow

    sigma_a = phi(sigma_v, r),   phi(u, s) = s * u / (1 + (s - 1) * u)

with r = `action_shift_ratio`. Shifts compose as a group
(phi(phi(u, s1), s2) = phi(u, s1 * s2)), so the action tokens effectively run
shift `video_shift * r` end to end while sharing the base clock u with the
video stream, and r = 1.0 degenerates exactly to `ImageWAMActionPatch`. The
per-token noise level reaches the backbone through `timestep_per_token`
(FLUX.2 Modulation/LastLayer broadcast a [B, N, hidden] vec token-wise; the
legacy scalar path is bit-identical when the argument is omitted).

Coordinates, codec, masks and the per-sample loss weight are unchanged from
the baseline; the weight stays a function of the *video* timestep for both
streams, so the only variable is the action noise level (and its conditioning).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from imagewam.utils.logging_config import get_logger

from exploration.action_img_patch.model import FLUX2_TOKEN_DIM, ImageWAMActionPatch

logger = get_logger(__name__)


class ImageWAMActionPatchSigmaShift(ImageWAMActionPatch):
    """Action tokens run their own (relative) sigma shift inside the joint denoise."""

    action_shift_ratio: float = 1.0

    def configure_action_shift(self, ratio: float) -> None:
        ratio = float(ratio)
        if ratio <= 0:
            raise ValueError(f"`action_shift_ratio` must be positive, got {ratio}")
        self.action_shift_ratio = ratio
        logger.info(
            "ImageWAMActionPatchSigmaShift: action_shift_ratio=%.4f "
            "(video train shift %.2f -> effective action shift %.3f)",
            ratio,
            float(self.train_video_scheduler.shift),
            float(self.train_video_scheduler.shift) * ratio,
        )

    @staticmethod
    def _shift_sigma(sigma: torch.Tensor, ratio: float) -> torch.Tensor:
        sigma32 = sigma.to(torch.float32)
        shifted = ratio * sigma32 / (1.0 + (ratio - 1.0) * sigma32)
        return shifted.to(dtype=sigma.dtype)

    # ---------------------------------------------------------- training loss
    def _training_loss_flux2(self, sample, tiled: bool = False):
        inputs = self.build_inputs_flux2(sample, tiled=tiled)
        target_latent = inputs["target_latent"]  # [B, S, 128]
        action = inputs["action"]                # [B, H, A]
        batch_size = int(target_latent.shape[0])
        target_len = int(target_latent.shape[1])
        horizon = int(action.shape[1])

        action_latent = self._encode_action_tokens(action)  # [B, H, 128]

        # Video sigma is sampled exactly as in the baseline (same RNG stream);
        # the action sigma is a deterministic Moebius reparametrization of it.
        scheduler = self.train_video_scheduler
        timestep = scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_latent.dtype,
        )
        num_train = float(scheduler.num_train_timesteps)
        sigma_video = timestep / num_train
        sigma_action = self._shift_sigma(sigma_video, self.action_shift_ratio)
        timestep_action = sigma_action * num_train

        noise_img = torch.randn_like(target_latent)
        noise_act = torch.randn_like(action_latent)
        noisy_img = scheduler.add_noise(target_latent, noise_img, timestep)
        noisy_act = scheduler.add_noise(action_latent, noise_act, timestep_action)
        target_v_img = scheduler.training_target(target_latent, noise_img, timestep)
        target_v_act = scheduler.training_target(action_latent, noise_act, timestep_action)

        target_img_ids = inputs["target_img_ids"]
        action_ids = self._build_action_token_ids(
            batch_size,
            horizon,
            device=target_img_ids.device,
            dtype=target_img_ids.dtype,
        )
        combined = torch.cat([noisy_img, noisy_act], dim=1)
        combined_ids = torch.cat([target_img_ids, action_ids], dim=1)

        # Per-token unit-domain timesteps over the img stream [ref | target | action].
        unit_video = self._scheduler_timestep_to_unit(timestep, scheduler)
        unit_action = self._scheduler_timestep_to_unit(timestep_action, scheduler)
        ref_latents = inputs["ref_image_latents"]
        cond_len = 0 if ref_latents is None else int(ref_latents.shape[1])
        timestep_per_token = torch.cat(
            [
                unit_video[:, None].expand(batch_size, cond_len + target_len),
                unit_action[:, None].expand(batch_size, horizon),
            ],
            dim=1,
        )

        video_pre = self.video_expert.pre_dit(
            x=combined,
            timestep=unit_video,
            context=inputs["text_hidden_states"],
            context_mask=inputs["text_attention_mask"],
            ref_image_hidden_states=ref_latents,
            target_img_ids=combined_ids,
            ref_img_ids=inputs["ref_img_ids"],
            timestep_per_token=timestep_per_token,
        )
        attention_mask = self._build_mot_attention_mask_flux2(
            batch_size=batch_size,
            txt_len=int(video_pre["txt_len"]),
            target_len=int(video_pre["target_len"]),
            cond_len=int(video_pre["cond_len"]),
            action_len=0,
            device=combined.device,
            text_attention_mask=video_pre["text_mask"],
        )
        tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
        pred = self.video_expert.post_dit(tokens_out, video_pre)  # [B, S+H, 128]
        pred_img = pred[:, :target_len]
        pred_act = pred[:, target_len:]

        weight = scheduler.training_weight(timestep)

        video_loss_per_sample = (
            F.mse_loss(pred_img.float(), target_v_img.float(), reduction="none").flatten(1).mean(dim=1)
        )
        video_weight = weight.to(video_loss_per_sample.device, dtype=video_loss_per_sample.dtype)
        loss_video = (video_loss_per_sample * video_weight).mean()

        action_loss_per_sample = self._compute_action_loss_per_sample(
            pred_action=pred_act,
            target_action=target_v_act,
            action_is_pad=inputs["action_is_pad"],
            action_dim_is_pad=None,
        )
        action_weight = weight.to(action_loss_per_sample.device, dtype=action_loss_per_sample.dtype)
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def infer_action_flux2(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> dict[str, Any]:
        """Baseline joint denoise, except the action rows walk their own sigma ladder."""
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(f"`input_image` must be [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}")

        text_hidden, text_mask = self._prepare_flux2_infer_text(prompt, context, context_mask)
        if self.proprio_encoder is not None or proprio is not None:
            text_hidden, text_mask = self._append_proprio_to_context_if_enabled(
                context=text_hidden,
                context_mask=text_mask,
                proprio=proprio,
                source="action-patch sigma-shift inference",
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        ref_tokens, ref_img_ids = self._encode_flux2_image_tokens(input_image, time_value=10.0)
        batch_size = int(ref_tokens.shape[0])
        target_len = int(ref_tokens.shape[1])
        cond_len = int(ref_tokens.shape[1])
        horizon = int(action_horizon)

        target_img_ids = ref_img_ids.clone()
        target_img_ids[..., 0] = 0.0
        action_ids = self._build_action_token_ids(
            batch_size,
            horizon,
            device=ref_img_ids.device,
            dtype=ref_img_ids.dtype,
        )
        combined_ids = torch.cat([target_img_ids, action_ids], dim=1)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device)
            generator.manual_seed(int(seed))
        latents = torch.randn(
            (batch_size, target_len + horizon, FLUX2_TOKEN_DIM),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        scheduler = self.infer_video_scheduler
        timesteps, deltas = scheduler.build_inference_schedule(
            num_inference_steps=int(num_inference_steps),
            device=self.device,
            dtype=self.torch_dtype,
            shift_override=sigma_shift,
        )
        # Reconstruct the full video sigma ladder (terminal 0 included), map it
        # through the relative shift and take pairwise differences: the action
        # Euler deltas paired step-for-step with the video ones.
        num_train = float(getattr(scheduler, "num_train_timesteps", 1000))
        sigma_video_full = torch.cat(
            [
                timesteps.to(torch.float32) / num_train,
                torch.zeros(1, device=timesteps.device, dtype=torch.float32),
            ]
        )
        sigma_action_full = self._shift_sigma(sigma_video_full, self.action_shift_ratio)
        action_deltas = sigma_action_full[1:] - sigma_action_full[:-1]

        attention_mask = None
        for step_index, (step_t, step_delta) in enumerate(zip(timesteps, deltas)):
            timestep = step_t.expand(batch_size).to(device=self.device, dtype=latents.dtype)
            unit_video = self._scheduler_timestep_to_unit(timestep, scheduler)
            unit_action = torch.full(
                (batch_size,),
                float(sigma_action_full[step_index]),
                device=self.device,
                dtype=latents.dtype,
            )
            timestep_per_token = torch.cat(
                [
                    unit_video[:, None].expand(batch_size, cond_len + target_len),
                    unit_action[:, None].expand(batch_size, horizon),
                ],
                dim=1,
            )
            video_pre = self.video_expert.pre_dit(
                x=latents,
                timestep=unit_video,
                context=text_hidden,
                context_mask=text_mask,
                ref_image_hidden_states=ref_tokens,
                target_img_ids=combined_ids,
                ref_img_ids=ref_img_ids,
                timestep_per_token=timestep_per_token,
            )
            if attention_mask is None:
                attention_mask = self._build_mot_attention_mask_flux2(
                    batch_size=batch_size,
                    txt_len=int(video_pre["txt_len"]),
                    target_len=int(video_pre["target_len"]),
                    cond_len=int(video_pre["cond_len"]),
                    action_len=0,
                    device=self.device,
                    text_attention_mask=video_pre["text_mask"],
                )
            tokens_out = self._forward_flux2_video_only(video_pre, attention_mask)
            pred = self.video_expert.post_dit(tokens_out, video_pre)
            pred_img = pred[:, :target_len]
            pred_act = pred[:, target_len:]
            latents = torch.cat(
                [
                    scheduler.step(pred_img, step_delta, latents[:, :target_len]),
                    scheduler.step(pred_act, action_deltas[step_index], latents[:, target_len:]),
                ],
                dim=1,
            )

        action = self._decode_action_tokens(latents[:, target_len:])
        return {"action": action[0].detach().to(device="cpu", dtype=torch.float32)}
