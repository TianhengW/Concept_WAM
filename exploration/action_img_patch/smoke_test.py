"""Smoke test for exploration/action_img_patch (1 GPU).

Checks, per model (--mode actionpatch | baseline | both):
  1. model builds with real weights;
  2. training_loss on a synthetic batch runs forward+backward without NaN;
  3. gradients reach the FLUX transformer (trainer freeze policy applied first);
  4. (actionpatch) fixed codec round-trip is exact and infer_action produces
     the right shape.

Run on a compute node:
  srun -p yukaichenglab -N1 --gres=gpu:1 --cpus-per-task=16 \
    bash -c 'cd /storage/yukaichengLab/mazijian/wth/ImageWAM && source .venv/bin/activate && \
      PYTHONPATH=$PWD:$PWD/src:/storage/yukaichengLab/mazijian/wth/flux2/src:/storage/yukaichengLab/mazijian/wth/flux2 \
      python exploration/action_img_patch/smoke_test.py --mode both'
"""

from __future__ import annotations

import argparse

import torch

FLUX2_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
AE_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
FLUX2_SRC = "/storage/yukaichengLab/mazijian/wth/flux2"
QWEN3 = "/storage/yukaichengLab/share/model/Qwen3-4B"


def make_sample(device: torch.device) -> dict:
    torch.manual_seed(0)
    return {
        # robotwin concat layout: 3 cams composed into one 288x256 frame
        "video": torch.rand(1, 3, 2, 288, 256, device=device),
        "action": torch.randn(1, 16, 14, device=device),
        "proprio": torch.randn(1, 14, device=device),
        "instruction": ["pick up the bottle and place it on the shelf"],
        "action_is_pad": torch.zeros(1, 16, dtype=torch.bool, device=device),
    }


def apply_trainer_freeze(model) -> None:
    # mirror Wan22Trainer policy: freeze all, train dit(+proprio), then refine
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    if getattr(model, "proprio_encoder", None) is not None:
        model.proprio_encoder.train()
        model.proprio_encoder.requires_grad_(True)
    policy = getattr(model, "apply_trainable_policy", None)
    if callable(policy):
        policy()


def check_backward(model, tag: str) -> None:
    device = model.device
    sample = make_sample(device)
    loss, logs = model.training_loss(sample)
    assert torch.isfinite(loss), f"[{tag}] loss is not finite: {loss}"
    loss.backward()
    grads = [
        p.grad.abs().sum().item()
        for p in model.video_expert.transformer.parameters()
        if p.grad is not None
    ]
    n_with_grad = sum(1 for g in grads if g > 0)
    assert n_with_grad > 0, f"[{tag}] no gradient reached the FLUX transformer"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{tag}] OK loss={float(loss):.4f} logs={logs} "
          f"transformer-params-with-grad={n_with_grad} trainable-params={trainable/1e9:.2f}B")
    model.zero_grad(set_to_none=True)


def smoke_actionpatch() -> None:
    from exploration.action_img_patch.model import ImageWAMActionPatch

    model = ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL,
        ae_model_path=AE_MODEL,
        flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3,
        qwen_context_len=128,
        proprio_dim=14,
        load_text_encoder=True,
        device="cuda",
        torch_dtype=torch.bfloat16,
        action_patch_dim=14,
    )
    # codec round-trip must be exact (fixed tiling + averaging)
    a = torch.randn(2, 16, 14, device=model.device, dtype=torch.float32)
    rt = model._decode_action_tokens(model._encode_action_tokens(a))
    err = (rt - a).abs().max().item()
    assert err < 1e-5, f"codec round-trip error {err}"
    print(f"[actionpatch] codec round-trip max-err={err:.2e}")

    apply_trainer_freeze(model)
    check_backward(model, "actionpatch")

    with torch.no_grad():
        out = model.infer_action(
            prompt="pick up the bottle",
            input_image=torch.rand(1, 3, 288, 256),
            action_horizon=16,
            proprio=torch.randn(1, 14),
            num_inference_steps=2,
        )
    assert out["action"].shape == (16, 14), f"infer_action shape {tuple(out['action'].shape)}"
    print(f"[actionpatch] infer_action OK shape={tuple(out['action'].shape)}")


def smoke_baseline() -> None:
    from imagewam.runtime import create_imagewam_flux2_klein

    model = create_imagewam_flux2_klein(
        flux2_model_path=FLUX2_MODEL,
        ae_model_path=AE_MODEL,
        flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3,
        qwen_context_len=128,
        load_text_encoder=True,
        proprio_dim=14,
        action_dit_config={
            "action_dim": 14,
            "hidden_dim": 1024,
            "mlp_ratio": 4.0,
            "max_action_horizon": 64,
            "use_gradient_checkpointing": True,
        },
        action_scheduler={"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
        loss={"lambda_video": 0.5, "lambda_action": 1.0},
        concept_k=0,
        lambda_concept=0.0,
        model_dtype=torch.bfloat16,
        device="cuda",
    )
    apply_trainer_freeze(model)
    check_backward(model, "baseline")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["actionpatch", "baseline", "both"], default="both")
    args = parser.parse_args()
    if args.mode in ("actionpatch", "both"):
        smoke_actionpatch()
        torch.cuda.empty_cache()
    if args.mode in ("baseline", "both"):
        smoke_baseline()
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
