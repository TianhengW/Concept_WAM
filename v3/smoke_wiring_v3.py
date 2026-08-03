"""Fast wiring smoke for v3 overlay: t0 tensor -> PIL -> VL reasoner -> [B,K,7680].

Validates the NEW overlay logic (ImageWAMV3._t0_to_pil + reasoner call) without
building the full FLUX.2 base model. The full training-step smoke goes through the
hydra config separately.

  PYTHONPATH=v3:src:$FLUX2/src:$FLUX2 .venv_v3/bin/python v3/smoke_wiring_v3.py
"""
import torch

VL_PATH = "/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct"


def main():
    from imagewam_v3 import VLLatentReasoner, ImageWAMV3

    B, C, T, H, W = 2, 3, 2, 256, 256
    K = 16
    dev = "cuda"

    # a fake batch's video: [B,C,T,H,W], RGB, normalized to [-1,1] (as the dataset does)
    video = (torch.rand(B, C, T, H, W) * 2 - 1).clamp(-1, 1)

    # 1) t0 -> PIL conversion (the overlay's static helper)
    pil = ImageWAMV3._t0_to_pil(video)
    assert len(pil) == B, len(pil)
    assert pil[0].size == (W, H) and pil[0].mode == "RGB", (pil[0].size, pil[0].mode)
    print(f"t0->PIL ok: {len(pil)} imgs, size={pil[0].size} mode={pil[0].mode}")

    # 2) reasoner accepts them and returns Plan-A conditioning
    reasoner = VLLatentReasoner(
        VL_PATH, mode="latent_loop", num_latent=K, condition_tokens="reasoning",
        freeze_vision=True, torch_dtype=torch.bfloat16,
    ).to(dev)
    reasoner.train()

    prompts = ["Pick up the red block.", "Open the drawer."]
    text_hidden, mask = reasoner(prompts, pil, device=dev)
    print(f"reasoner out: text_hidden {tuple(text_hidden.shape)} mask {tuple(mask.shape)} "
          f"sum={mask.sum().item()}")
    assert text_hidden.shape == (B, K, 7680), text_hidden.shape
    assert mask.shape == (B, K) and mask.all()
    assert torch.isfinite(text_hidden).all()

    # 3) gradient flows (full-param LLM) through the whole t0-grounded path
    loss = text_hidden.float().pow(2).mean()
    loss.backward()
    llm_grad = sum(
        float(p.grad.abs().sum()) for n, p in reasoner.named_parameters()
        if "language_model" in n and p.grad is not None
    )
    assert llm_grad > 0, "no grad reached the LLM"
    print(f"backward ok: llm_grad={llm_grad:.3e}")
    print("WIRING SMOKE OK")


if __name__ == "__main__":
    main()
