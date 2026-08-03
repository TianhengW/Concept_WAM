# ImageWAM v2 — Arm A (Qwen3-4B as a K-slot generator)

Goal: replace the full-length Qwen3 text features with **K = 16 conditioning
embeddings** produced by Qwen3-4B in a single forward (16 learnable query slots),
and fine-tune Qwen3-4B **fully** (no LoRA) end-to-end with the MoT image/action
flow-matching loss.

This is the "implicit reasoning" arm: no explicit chain-of-thought text is
emitted; the reasoning, if any, stays inside Qwen's forward pass.

## Layout
- `imagewam_v2/qwen_slot_generator.py` — the K-slot generator module.
- `imagewam_v2/model_v2.py` — `ImageWAMV2`, overrides only the FLUX.2 text path.
- `imagewam_v2/runtime_v2.py` — `create_imagewam_v2`, the hydra model target.
- `configs/robotwin_v2_armA.yaml` — self-contained training config.
- `scripts/env_v2.sh` — shared paths / env.
- `scripts/launch_v2.sh` — accelerate launcher (single- or multi-node).
- `scripts/train_v2.py` — hydra entry -> `run_training`.
- `scripts/smoke_v2.sbatch` — 1-node, 6-step smoke.
- `scripts/sbatch_v2_armA_4node.sh` — 4-node training.

## How it plugs in without editing the base package
- The generator is attached as `mot.slot_generator`, so the base trainer's
  `model.dit.parameters()` optimizer and `mot.state_dict()` checkpointing cover
  it automatically.
- `ImageWAMV2._encode_flux2_text` (no `no_grad`) runs the generator so gradients
  reach Qwen and the query embeddings.
- Resume weights from the released FLUX.2-4B RoboTwin checkpoint (MoT loads with
  `strict=False`; the generator keeps its Qwen init).

## Run
```bash
# smoke (1 node, 8 GPU)
sbatch v2/scripts/smoke_v2.sbatch
# full (4 nodes, 32 GPU)
sbatch v2/scripts/sbatch_v2_armA_4node.sh
```
