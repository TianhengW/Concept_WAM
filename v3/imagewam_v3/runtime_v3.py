"""Hydra entry point for building the v3 model (VL LatentReasoner).

`create_imagewam_v3` mirrors `imagewam.runtime.create_imagewam_flux2_klein`, builds
the base FLUX.2 model WITHOUT the Qwen text encoder (v3 uses a separate Qwen3-VL model,
not the base Qwen3-4B), then attaches the image-grounded VL Coconut reasoner.
"""
from __future__ import annotations

import torch

from imagewam.runtime import create_imagewam_flux2_klein
from imagewam.utils.logging_config import get_logger

from .model_v3 import ImageWAMV3
from .vl_latent_reasoner import VLLatentReasoner

logger = get_logger(__name__)


def create_imagewam_v3(
    flux2_model_path: str,
    ae_model_path: str,
    vl_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    flux2_lora_config=None,
    concept_k: int = 0,
    lambda_concept: float = 0.0,
    # --- VL reasoner knobs ---
    reasoner_mode: str = "latent_loop",
    num_latent: int = 16,
    reasoner_layers=(9, 18, 27),
    condition_tokens: str = "reasoning",
    freeze_vision: bool = True,
    gradient_checkpointing: bool = False,
    reasoner_system_prompt: str | None = None,
    reasoner_max_prompt_len: int = 256,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    # v3 does not use the base Qwen3-4B text encoder; the VL reasoner replaces it.
    model = create_imagewam_flux2_klein(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=variant,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=proprio_dim,
        load_text_encoder=False,
        video_scheduler=video_scheduler,
        action_scheduler=action_scheduler,
        loss=loss,
        mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        mot_gqa_implementation=mot_gqa_implementation,
        mot_force_flash_attention=mot_force_flash_attention,
        pack_proprio_after_text=pack_proprio_after_text,
        flux2_lora_config=flux2_lora_config,
        model_dtype=model_dtype,
        device=device,
        concept_k=int(concept_k),
        lambda_concept=float(lambda_concept),
    )

    reasoner = VLLatentReasoner(
        vl_model_path,
        mode=str(reasoner_mode),
        num_latent=int(num_latent),
        qwen_layers=tuple(reasoner_layers),
        condition_tokens=str(condition_tokens),
        freeze_vision=bool(freeze_vision),
        gradient_checkpointing=bool(gradient_checkpointing),
        system_prompt=reasoner_system_prompt,
        max_prompt_len=int(reasoner_max_prompt_len),
        torch_dtype=model_dtype,
    ).to(device=device, dtype=model_dtype)

    model.__class__ = ImageWAMV3
    model.attach_reasoner(reasoner)

    n_trainable = sum(p.numel() for p in reasoner.parameters() if p.requires_grad)
    logger.info("v3 VL reasoner trainable params (full-param LLM): %.1fM", n_trainable / 1e6)
    return model
