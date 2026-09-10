"""Scheme A of the H3/TMRoPE study: time-aligned RoPE coordinates for action tokens.

`ImageWAMActionPatch` parks the action chunk on a separate categorical time
plane (axis0 = 20.0, chunk index on axis 3), so the rotary encoding tells the
model which group a token belongs to but not *when* it happens. Following
MiniMax-H3 (audio latents share the video's rotary clock) and Qwen2.5-Omni's
TMRoPE (absolute-time position ids across modalities), this variant places
every action token at its physical timestamp on the time axis instead: the
clean reference frame sits at 10.0 ("now"), the noisy target frame at 0.0
(`endpoint_frames_only` guarantees the 16-step chunk spans exactly
ref -> target), and action k -- which targets state[k+1], i.e. the transition
over (k, k+1] -- sits at the interval midpoint. Midpoints never hit 0.0/10.0
exactly, so action rows stay positionally disjoint from the text plane
(axis0 = 0) and both frame planes without any extra separator. Codec, losses,
masks and all video-token coordinates are bit-identical to the baseline.
"""

from __future__ import annotations

import torch

from exploration.action_img_patch.model import ImageWAMActionPatch

REF_FRAME_TIME_VALUE = 10.0
TARGET_FRAME_TIME_VALUE = 0.0


class ImageWAMActionPatchTimeAlign(ImageWAMActionPatch):
    """Action tokens live at their physical time between ref (10.0) and target (0.0)."""

    @staticmethod
    def _build_action_token_ids(
        batch_size: int,
        horizon: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        ids = torch.zeros(batch_size, horizon, 4, device=device, dtype=dtype)
        span = REF_FRAME_TIME_VALUE - TARGET_FRAME_TIME_VALUE
        midpoints = (torch.arange(horizon, device=device, dtype=dtype) + 0.5) / float(horizon)
        ids[..., 0] = (REF_FRAME_TIME_VALUE - midpoints * span)[None, :]
        ids[..., 3] = torch.arange(horizon, device=device, dtype=dtype)[None, :]
        return ids
