"""Smoke test: action-as-patch + frozen Qwen3-VL text encoder (1 GPU, .venv_v3).

Checks:
  1. VLPromptEncoder loads Qwen3-VL and encodes (image+text) -> [B, L, 7680];
  2. VL is fully frozen (0 trainable params);
  3. fixed action codec round-trip is exact (inherited from patch);
  4. training_loss forward+backward finite, grad reaches FLUX transformer,
     and VL receives NO gradient (frozen + no_grad feature extractor);
  5. infer_action produces the right shape.
"""
from __future__ import annotations
import torch

FLUX2_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
AE_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
FLUX2_SRC = "/storage/yukaichengLab/mazijian/wth/flux2"
VL = "/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct"


def make_sample(device):
    torch.manual_seed(0)
    return {
        "video": torch.rand(1, 3, 2, 288, 256, device=device) * 2 - 1,   # [-1,1]
        "action": torch.randn(1, 16, 14, device=device),
        "proprio": torch.randn(1, 14, device=device),
        "instruction": ["pick up the bottle and place it on the shelf"],
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool, device=device),
    }


def apply_trainer_freeze(model):
    model.requires_grad_(False)
    model.dit.train(); model.dit.requires_grad_(True)
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.train(); model.proprio_encoder.requires_grad_(True)
    policy = getattr(model, "apply_trainable_policy", None)
    if callable(policy):
        policy()


def main():
    from exploration.action_img_patch.factory import create_imagewam_flux2_klein_actionpatch_vl
    model = create_imagewam_flux2_klein_actionpatch_vl(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        vl_model_path=VL, vl_qwen_layers=[9, 18, 27], vl_max_prompt_len=128,
        proprio_dim=14, action_patch_dim=14, action_patch_scale=1.0,
        video_scheduler={"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
        loss={"lambda_video": 0.5, "lambda_action": 1.0},
        model_dtype=torch.bfloat16, device="cuda",
    )

    # 1. VL encode dim
    imgs = model._frame_to_pil(torch.rand(1, 3, 288, 256) * 2 - 1)
    th, mask = model.vl_encoder.encode(["pick up the bottle"], imgs)
    print(f"[1] VL encode hidden={tuple(th.shape)} mask={tuple(mask.shape)} (expect last dim 7680)")
    assert th.shape[-1] == 7680, f"VL out dim {th.shape[-1]} != 7680"

    # 2. VL frozen
    n_vl_train = sum(p.numel() for p in model.vl_encoder.vl.parameters() if p.requires_grad)
    assert n_vl_train == 0, f"VL not frozen: {n_vl_train} trainable"
    print(f"[2] VL frozen OK (0 trainable in VL)")

    # 3. codec round-trip
    a = torch.randn(2, 16, 14, device=model.device, dtype=torch.float32)
    rt = model._decode_action_tokens(model._encode_action_tokens(a))
    err = (rt - a).abs().max().item()
    assert err < 1e-5, f"codec round-trip err {err}"
    print(f"[3] codec round-trip max-err={err:.2e}")

    # 4. forward+backward
    apply_trainer_freeze(model)
    sample = make_sample(model.device)
    loss, logs = model.training_loss(sample)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    fg = [p.grad.abs().sum().item() for p in model.video_expert.transformer.parameters() if p.grad is not None]
    n_fg = sum(1 for g in fg if g > 0)
    assert n_fg > 0, "no gradient reached FLUX transformer"
    vl_g = [p for p in model.vl_encoder.vl.parameters() if p.grad is not None]
    assert len(vl_g) == 0, f"VL received gradient ({len(vl_g)}) — should be frozen+no_grad"
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[4] backward OK loss={float(loss):.4f} logs={logs} FLUX-grad-params={n_fg} "
          f"VL-grad-params={len(vl_g)} trainable={trn/1e9:.2f}B")
    model.zero_grad(set_to_none=True)

    # 5. infer_action
    with torch.no_grad():
        out = model.infer_action(prompt="pick up the bottle",
                                 input_image=torch.rand(1, 3, 288, 256),
                                 action_horizon=16, proprio=torch.randn(1, 14),
                                 num_inference_steps=2)
    assert out["action"].shape == (16, 14), f"infer shape {tuple(out['action'].shape)}"
    print(f"[5] infer_action OK shape={tuple(out['action'].shape)}")
    print("VL SMOKE PASSED")


if __name__ == "__main__":
    main()
