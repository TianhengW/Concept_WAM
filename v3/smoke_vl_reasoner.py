"""Standalone smoke test for VLLatentReasoner (v3).

Loads the real Qwen3-VL-4B weights, runs a forward on a batch of (t0 image, prompt),
checks the output contract [B, L+K, 7680] + mask, then a backward to confirm gradients
flow to LoRA / bot / feedback only and the vision tower stays frozen.

Run under .venv_v3:
  PYTHONPATH=v3:src:$FLUX2/src:$FLUX2 .venv_v3/bin/python v3/smoke_vl_reasoner.py
"""
import sys
import torch
from PIL import Image

VL_PATH = "/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct"


def main():
    from imagewam_v3 import VLLatentReasoner

    mode = sys.argv[1] if len(sys.argv) > 1 else "latent_loop"
    dev = "cuda"
    K = 16
    reasoner = VLLatentReasoner(
        VL_PATH,
        mode=mode,
        num_latent=K,
        freeze_vision=True,
        torch_dtype=torch.bfloat16,
    ).to(dev)

    # two synthetic t0 frames + instructions
    imgs = [
        Image.new("RGB", (224, 224), (120, 30, 60)),
        Image.new("RGB", (256, 192), (10, 90, 140)),
    ]
    prompts = [
        "Pick up the red block and place it on the plate.",
        "Open the drawer and put the mouse pad inside.",
    ]

    reasoner.train()
    text_hidden, mask = reasoner(prompts, imgs, device=dev)
    print(f"[{mode}] cond={reasoner.condition_tokens} text_hidden {tuple(text_hidden.shape)} "
          f"dtype={text_hidden.dtype}  mask {tuple(mask.shape)} sum={mask.sum().item()}")
    assert text_hidden.shape[0] == 2, text_hidden.shape
    assert text_hidden.shape[-1] == 7680, text_hidden.shape
    assert text_hidden.shape[1] == mask.shape[1]
    # Plan A default: exactly K reasoning tokens, all unmasked
    assert text_hidden.shape[1] == K, f"expected K={K} tokens, got {text_hidden.shape[1]}"
    assert mask.all(), "all reasoning positions must be unmasked"
    assert torch.isfinite(text_hidden).all(), "non-finite output"

    # backward through a dummy conditioning loss
    loss = text_hidden.float().pow(2).mean()
    loss.backward()

    grad_llm = grad_bot = grad_fb = grad_thought = 0.0
    llm_params_with_grad = 0
    vision_has_grad = False
    for n, p in reasoner.named_parameters():
        g = 0.0 if p.grad is None else float(p.grad.detach().abs().sum())
        if "visual" in n:
            if p.grad is not None:
                vision_has_grad = True
        elif "language_model" in n:
            grad_llm += g
            if p.grad is not None and g > 0:
                llm_params_with_grad += 1
        elif n.endswith("bot_embedding"):
            grad_bot += g
        elif "feedback" in n:
            grad_fb += g
        elif n.endswith("thought_embeddings"):
            grad_thought += g
    print(f"grad: llm={grad_llm:.3e} (#params>0={llm_params_with_grad}) bot={grad_bot:.3e} "
          f"feedback={grad_fb:.3e} thought={grad_thought:.3e}  vision_has_grad={vision_has_grad}")
    assert grad_llm > 0 and llm_params_with_grad > 100, "LLM full-param grads missing"
    if mode == "latent_loop":
        assert grad_bot > 0 and grad_fb > 0, "no gradient to bot/feedback"
    else:
        assert grad_thought > 0, "no gradient to thought embeddings"
    assert not vision_has_grad, "vision tower must stay frozen"

    n_train = sum(p.numel() for p in reasoner.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
