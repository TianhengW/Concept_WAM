"""Hydra factory for the action-as-patch single-stream model.

`_target_` entry: exploration.action_img_patch.factory.create_imagewam_flux2_klein_actionpatch
(the sbatch scripts prepend REPO_ROOT to PYTHONPATH so `exploration` is importable).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf


def _to_dict(value, name: str) -> dict:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}")
    return value


def create_imagewam_flux2_klein_actionpatch(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    action_attn_isolate: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from exploration.action_img_patch.model import ImageWAMActionPatch

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")

    return ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
        action_attn_isolate=bool(action_attn_isolate),
    )


def create_imagewam_flux2_klein_actionpatch_vl(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    vl_model_path: str = "/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct",
    vl_qwen_layers=(9, 18, 27),
    vl_max_prompt_len: int = 128,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """action-as-patch,文本编码器换成冻结的、图像 grounded 的 Qwen3-VL(看 t0 当前帧)。
    其余与 create_imagewam_flux2_klein_actionpatch 逐项一致(同一 from_pretrained、同参数)。"""
    from exploration.action_img_patch.model_vl import ImageWAMActionPatchVL, VLPromptEncoder

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")

    model = ImageWAMActionPatchVL.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=None,
        qwen_context_len=int(vl_max_prompt_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=False,          # 不加载 Qwen3-4B;text 全走 VL
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
    )
    enc = VLPromptEncoder(
        vl_model_path,
        qwen_layers=tuple(int(l) for l in vl_qwen_layers),
        max_prompt_len=int(vl_max_prompt_len),
        device=device,
        torch_dtype=model_dtype,
    )
    model.attach_vl_encoder(enc)
    return model


def create_imagewam_flux2_klein_actionrouter(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    router=None,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """action-as-patch 的 patch-routed 变体:动作查询 token 经 top-k router
    从图像 patch 表征聚合并回归动作;视频侧与 AP 完全一致。"""
    from exploration.action_img_patch.model_action_router import ImageWAMActionRouter

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")
    router = _to_dict(router, "router")

    model = ImageWAMActionRouter.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
    )
    model.configure_router(
        topk=int(router.get("topk", 8)),
        lambda_balance=float(router.get("lambda_balance", 1.0e-2)),
        lambda_entropy=float(router.get("lambda_entropy", 1.0e-2)),
        action_flow_steps=int(router.get("action_flow_steps", 10)),
    )
    return model


def create_imagewam_flux2_klein_actionpatch_timealign(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """方案 A(H3/TMRoPE 时间对齐):action token 的 RoPE 轴0 从类别平面(20.0)
    改为 ref(10.0)→target(0.0) 区间的物理时间插值(区间中点约定);
    其余与 create_imagewam_flux2_klein_actionpatch 逐项一致。"""
    from exploration.action_img_patch.model_timealign import ImageWAMActionPatchTimeAlign

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")

    return ImageWAMActionPatchTimeAlign.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
    )


def create_imagewam_flux2_klein_actionpatch_sigmashift(
    flux2_model_path: str,
    ae_model_path: str,
    flux2_src_path: str | None = None,
    variant: str = "klein-base-4b",
    qwen3_model_spec: str | None = None,
    qwen_context_len: int = 128,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    mot_checkpoint_mixed_attn: bool = True,
    mot_gqa_implementation: str = "repeat",
    mot_force_flash_attention: bool = False,
    pack_proprio_after_text: bool = True,
    video_scheduler=None,
    loss=None,
    action_patch_dim: int = 14,
    action_patch_scale: float = 1.0,
    action_shift_ratio: float = 0.25,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """方案 B(H3 双 schedule):action sigma = phi(video sigma, ratio),
    ratio=1.0 精确退化基线;其余与 create_imagewam_flux2_klein_actionpatch 逐项一致。"""
    from exploration.action_img_patch.model_sigmashift import ImageWAMActionPatchSigmaShift

    video_scheduler = _to_dict(video_scheduler, "video_scheduler")
    loss = _to_dict(loss, "loss")

    model = ImageWAMActionPatchSigmaShift.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=flux2_model_path,
        ae_model_path=ae_model_path,
        flux2_src_path=flux2_src_path,
        variant=str(variant),
        qwen3_model_spec=qwen3_model_spec,
        qwen_context_len=int(qwen_context_len),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        load_text_encoder=bool(load_text_encoder),
        device=device,
        torch_dtype=model_dtype,
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        mot_gqa_implementation=str(mot_gqa_implementation),
        mot_force_flash_attention=bool(mot_force_flash_attention),
        pack_proprio_after_text=bool(pack_proprio_after_text),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        loss_lambda_video=float(loss.get("lambda_video", 0.5)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_patch_dim=int(action_patch_dim),
        action_patch_scale=float(action_patch_scale),
    )
    model.configure_action_shift(float(action_shift_ratio))
    return model
