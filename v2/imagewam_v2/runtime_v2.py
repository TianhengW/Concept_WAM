"""Hydra entry point for building the Arm A model.

`create_imagewam_v2` mirrors `imagewam.runtime.create_imagewam_flux2_klein`,
builds the base FLUX.2 model (with the Qwen text encoder loaded), then attaches
the K-slot generator that reuses the already-loaded Qwen (no second copy).
"""
from __future__ import annotations

import torch

from imagewam.runtime import create_imagewam_flux2_klein
from imagewam.utils.logging_config import get_logger

from .model_v2 import ImageWAMV2
from .qwen_slot_generator import QwenSlotGenerator

logger = get_logger(__name__)


def create_imagewam_v2(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 512,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    proprio_dim: int | None = None,
    load_text_encoder: bool = True,
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
    # --- Arm A slot generator knobs ---
    slot_k: int = 16,
    slot_output_layers=(9, 18, 27),
    slot_max_length: int = 512,
    slot_gradient_checkpointing: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    if not bool(load_text_encoder):
        raise ValueError("Arm A requires load_text_encoder=true (online Qwen forward with gradients).")

    model = create_imagewam_flux2_klein(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=variant,
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=qwen_context_len,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        proprio_dim=proprio_dim,
        load_text_encoder=True,
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

    if model.text_encoder is None or not hasattr(model.text_encoder, "model"):
        raise RuntimeError("Expected a loaded Qwen text encoder to build the slot generator.")

    # inner Qwen3Model (decoder stack without lm_head): reuse the already-loaded
    # weights so we do not hold a second 4B copy.
    causal_lm = model.text_encoder.model
    qwen_base = getattr(causal_lm, "model", causal_lm)
    tokenizer = model.text_encoder.tokenizer

    slot_generator = QwenSlotGenerator(
        qwen_base,
        tokenizer,
        K=int(slot_k),
        output_layers=tuple(slot_output_layers),
        max_length=int(slot_max_length),
        gradient_checkpointing=bool(slot_gradient_checkpointing),
    ).to(device=device, dtype=model_dtype)

    # Promote to the v2 subclass (same instance, no re-init) and attach.
    model.__class__ = ImageWAMV2
    model.attach_slot_generator(slot_generator)

    n_trainable = sum(p.numel() for p in slot_generator.parameters())
    logger.info("Arm A slot generator params (all trainable under full fine-tune): %.1fM", n_trainable / 1e6)
    return model
