"""Inference latency benchmark: MoT base vs action-as-patch, same GPU, same input.

Measures end-to-end infer_action latency (action chunk generation), which is
what closed-loop control cares about. Loads the two models sequentially in one
process (frees the first before the second).
"""

import time

import torch

FLUX2_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
AE_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
FLUX2_SRC = "/storage/yukaichengLab/mazijian/wth/flux2"
QWEN3 = "/storage/yukaichengLab/share/model/Qwen3-4B"

N_WARMUP = 2
N_RUNS = 10
STEPS = 20
HORIZON = 16


def bench(model, tag):
    img = torch.rand(1, 3, 288, 256)
    proprio = torch.randn(1, 14)
    times = []
    for i in range(N_WARMUP + N_RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.infer_action(
            prompt="pick up the bottle and place it on the shelf",
            input_image=img,
            action_horizon=HORIZON,
            proprio=proprio,
            num_inference_steps=STEPS,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= N_WARMUP:
            times.append(dt)
    assert out["action"].shape == (HORIZON, 14)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print("[%s] infer_action %d steps: mean %.3fs  std %.3fs  min %.3fs  max %.3fs  (n=%d)"
          % (tag, STEPS, mean, std, min(times), max(times), len(times)), flush=True)
    return mean


def load_base():
    from imagewam.runtime import create_imagewam_flux2_klein
    return create_imagewam_flux2_klein(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, load_text_encoder=True, proprio_dim=14,
        action_dit_config={"action_dim": 14, "hidden_dim": 1024, "mlp_ratio": 4.0,
                           "max_action_horizon": 64, "use_gradient_checkpointing": False},
        action_scheduler={"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
        loss={"lambda_video": 0.5, "lambda_action": 1.0},
        concept_k=0, lambda_concept=0.0,
        model_dtype=torch.bfloat16, device="cuda",
    )


def load_patch():
    from exploration.action_img_patch.model import ImageWAMActionPatch
    return ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
        flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
        qwen3_model_spec=QWEN3, qwen_context_len=128, proprio_dim=14, load_text_encoder=True,
        device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14,
    )


def main():
    m = load_base()
    m.eval()
    t_base = bench(m, "base (MoT: video prefill + KV-cache action-only decode)")
    del m
    torch.cuda.empty_cache()

    m = load_patch()
    m.eval()
    t_patch = bench(m, "patch (single-stream joint denoise, full sequence)")
    del m
    torch.cuda.empty_cache()

    print("RESULT: base %.3fs vs patch %.3fs  -> patch/base = %.2fx" % (t_base, t_patch, t_patch / t_base))


if __name__ == "__main__":
    main()
