# ImageWAM

Official codebase for **ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?**

[English](./README.md) | [中文](./README_zh.md) 

[Huggingface Models](https://huggingface.co/collections/yuyangalin/imagewam) | [Paper](https://arxiv.org/abs/2606.19531) | [Project Page](https://zhangwenyao1.github.io/ImageWAM/)

ImageWAM is a family of world action models built on image-editing foundation models. This repository contains the training and evaluation code used in the paper experiments on LIBERO, LIBERO-plus, and RoboTwin.

We recommend starting with **FLUX.2 ImageWAM**. It provides 4B and 9B variants based on FLUX.2 [klein] 4B/9B base models, and gives the strongest performance in the series. This repository also provides training and evaluation entrypoints for **OmniGen2 ImageWAM** and **Ovis-U1 ImageWAM**. These variants are built on OmniGen2 and Ovis-U1 and also perform well. The Ovis-U1 variant is the smallest model in the series, with only a 1.1B DiT for image editing, while remaining competitive with larger variants in many settings.

All commands below are assumed to run from the repository root.

## Table Of Contents

- [Repository Structure](#repository-structure)
- [Three-Stream + Latent WAM](#three-stream--latent-wam)
- [Action-as-Patch (AP)](#action-as-patch-ap)
- [Basic Installation](#basic-installation)
- [Model Preparation](#model-preparation)
- [Data Preparation](#data-preparation)
- [Benchmark Environments](#benchmark-environments)
- [Training](#training)
- [Evaluation](#evaluation)
- [Release Checkpoints](#release-checkpoints)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Repository Structure

```text
ImageWAM/
├── configs/                  # Model, data, task, and benchmark configs
├── docs/                     # More detailed setup, data, model, and dependency notes
├── experiments/              # LIBERO / RoboTwin evaluation managers
├── scripts/
│   ├── flux2/                # FLUX.2 ImageWAM training and evaluation entrypoints
│   ├── omnigen2/             # OmniGen2 ImageWAM training and evaluation entrypoints
│   ├── ovis_u1/              # Ovis-U1 ImageWAM training and evaluation entrypoints
│   ├── data/                 # Data processing utilities
│   └── setup/                # Benchmark environment setup helpers
├── src/imagewam/             # Core ImageWAM code
└── third_party/              # Vendored benchmark / model adapter code
```

You also need to prepare datasets locally, usually under `./data`, pretrained model weights, and generated ActionDiT initialization weights, usually under `./checkpoints`.

## Three-Stream + Latent WAM

This branch (`latent`) adds a **three-stream (image + text + action) VLA** with an online
**LatentReasoner** on top of the FLUX.2-Klein-4B backbone, trained on RoboTwin **with the
exact same data setting as the released ImageWAM model** (non-idle / no-op frame filtering,
see [RoboTwin data preparation](#robotwin)).

### Architecture

- **Three-stream core** — `src/imagewam/models/backbones/flux2_action_three_stream.py`
  (`Flux2ActionTransformer2DModel`): the image and text streams keep the exact FLUX.2
  double/single blocks (pretrained weights load unchanged), and a bottlenecked **action
  expert** joins the joint attention in every layer. An ImageWAM visibility mask enforces
  that action attends only `text(+state) + clean context + itself`, nothing attends the
  action tokens, and padded prompt tokens are masked out.
- **LatentReasoner** — `src/imagewam/models/backbones/latent_reasoner.py`: one Qwen3-4B
  forward yields the prompt representations (layers 9/18/27 concatenated → 7680-d, matching
  `joint_attention_dim`) **plus `num_latent`=16 appended Coconut-style latent reasoning
  tokens**. Its output is fed directly as the three-stream `encoder_hidden_states` (the text
  stream), so the reasoning conditions the action expert.
- **Top-level model + loss** — `src/imagewam/models/backbones/flux2_three_stream_model.py`
  (`ImageWAMThreeStream`): VAE-encodes the target/context frames, runs the reasoner, adds
  **independent** video/action flow-matching noise, and returns the dual loss
  (`loss_video` + `loss_action`).
- **Runtime factory** — `imagewam.runtime.create_imagewam_flux2_klein_threestream`.

### Data (must match the release setting)

Follow [RoboTwin data preparation](#robotwin) to download `robotwin2.0-fastwam` and, most
importantly, generate the **non-idle (no-op) filter** — the dataloader restricts sampling to
the kept ranges (~89.3% of frames), identical to the release recipe:

```bash
ROBOTWIN_ROOT=/path/to/robotwin2.0 bash scripts/data/precompute_noops_lerobot.sh
# -> ${ROBOTWIN_ROOT}/nonidle_ranges.json   (default OpenPI-DROID recipe: idle_l2=1e-3, min_idle_len=5)
```

The task config `configs/task/robotwin_flux2_klein_4b_threestream_latent_imagewam.yaml`
already points `data.{train,val}.nonidle_filter_path` at this file, with `num_frames=17`,
`endpoint_frames_only=true`, `compact_288x256` cameras and z-score normalization from the
dataset's own `dataset_stats.json` — matching the release exactly.

### Training

```bash
# 4 nodes x 8 GPUs = 32, ZeRO-1, global batch 256 (batch 4 x accum 2), 10 epochs
sbatch scripts/flux2/sbatch_threestream_latent_4node.sh

# single-node smoke (8 GPUs, 2 steps)
TASK=robotwin_flux2_klein_4b_threestream_latent_imagewam FLUX2_SRC=/path/to/flux2 \
  bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
  num_epochs=1 max_steps=2 batch_size=1 gradient_accumulation_steps=1 wandb.enabled=false
```

Adjust the SBATCH partition / node names and the `NCCL_SOCKET_IFNAME` / rendezvous lines in
the 4-node launcher for your cluster; point the FLUX.2 transformer / AE / Qwen3 weights via
hydra overrides (`model.flux2_transformer_dir=... model.ae_model_path=...
model.qwen3_model_spec=...`) or edit
`configs/model/imagewam_flux2_klein_4b_threestream_latent.yaml`.

### Key files

| Purpose | Path |
|---|---|
| Three-stream transformer | `src/imagewam/models/backbones/flux2_action_three_stream.py` |
| LatentReasoner (Qwen3 Coconut) | `src/imagewam/models/backbones/latent_reasoner.py` |
| Top-level model + dual loss | `src/imagewam/models/backbones/flux2_three_stream_model.py` |
| Runtime factory | `src/imagewam/runtime.py` → `create_imagewam_flux2_klein_threestream` |
| Model / task configs | `configs/model/imagewam_flux2_klein_4b_threestream_latent.yaml` · `configs/task/robotwin_flux2_klein_4b_threestream_latent_imagewam.yaml` |
| Non-idle filter generator | `scripts/data/precompute_noops_lerobot.sh` · `scripts/data/compute_robotwin_nonidle_ranges.py` |
| 4-node training launcher | `scripts/flux2/sbatch_threestream_latent_4node.sh` |
| Train entry / accelerate | `scripts/train.py` · `scripts/flux2/train_flux2_klein_imagewam.sh` · `scripts/train_zero1.sh` |


## Action-as-Patch (AP)

Action-as-patch keeps the **single-stream** FLUX.2 backbone and injects actions as extra
latent *patches* instead of adding a separate action expert. An action chunk of
`action_horizon` steps is tiled into 128-d latent tokens, concatenated with the video
tokens, and denoised by the **same** transformer under **one** sigma. The loss is
`0.5 * loss_video + 1.0 * loss_action`.

Because there is no second tower, AP adds almost no parameters over the base ImageWAM
model, and video and action stay tightly coupled through shared attention.

### Architecture

| | |
|---|---|
| Backbone | FLUX.2-klein-4B (single stream, shared for video + action) |
| Action injection | chunk `[T, action_dim]` → tiled into 128-d latent patches → concatenated with video tokens |
| Timestep | **one** sigma for video and action (no separate action schedule) |
| Loss | `0.5 * loss_video + 1.0 * loss_action` |
| Variants | `actionpatch` (default) · `_isoattn` · `_sigmashift` · `_timealign` · `_vl` · `actionrouter` |

### Data (RoboDojo)

RoboDojo ships as raw HDF5; convert it to the LeRobot layout, then precompute the
normalization stats and the non-idle ranges:

```bash
# 1) raw HDF5 -> LeRobot (parquet + per-camera mp4)
bash scripts/download_robodojo_hdf5.sh                 # or download_robodojo_lerobot_share.sh
python scripts/data/robodojo_raw_to_lerobot.py --src <hdf5_root> --dst <lerobot_root>

# 2) normalization stats (action / state mean+std, per-step stats)
python scripts/data/compute_robodojo_dataset_stats.py --root <lerobot_root>
# -> <lerobot_root>/dataset_stats.json

# 3) non-idle ranges (drops idle prefixes/suffixes; same recipe as RoboTwin)
#    -> <lerobot_root>/nonidle_ranges.json
```

> **Re-encode the videos before training.** RoboDojo's released mp4s are AV1 with a single
> keyframe per clip, so a random seek decodes hundreds of frames to return 17. On one
> 2-GPU host this held the GPUs at ~19% duty (225 W of a 700 W TDP) and 7.3 s/step.
> Transcoding to H.264 with `GOP=25` cut per-camera decode from 284 ms to 28 ms and took
> the same job to 1.3 s/step at 648 W. See the transcode helper in `scripts/data/`.

The task config points at two roots (`RoboDojo_lerobot` and `RoboDojo_lerobot_part2`,
split only by download batch) and shares one pooled `dataset_stats.json`.

### Training

```bash
# 4 nodes x 8 GPUs = 32, ZeRO-1, global batch 256 (batch 4 x accum 2), 16 epochs
sbatch exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh

# resume from a saved optimizer state (NOT the weights/*.pt file — see note below)
sbatch exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh \
  resume=runs/robodojo_flux2_klein_4b_actionpatch_full/<timestamp>/checkpoints/state/step_080000

# single-node smoke (8 GPUs, 2 steps) — no Slurm
TASK=robodojo_flux2_klein_4b_actionpatch_full FLUX2_SRC=/path/to/flux2 \
  bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
  num_epochs=1 max_steps=2 batch_size=1 gradient_accumulation_steps=1 wandb.enabled=false

# RoboTwin instead of RoboDojo — same recipe, different TASK
sbatch exploration/action_img_patch/sbatch_actionpatch_full_4node.sh
```

Before the first run, edit these in the launcher for your cluster:

| Line | What to change |
|---|---|
| `#SBATCH -p` / `-N` / `--gres` | partition, node count, GPUs per node |
| `#SBATCH -x` | node exclusion list (see *Operational notes*) |
| `REPO_ROOT=` | absolute path to this repo |
| `FLUX2_SRC=` | path to the `flux2` source tree (the AP repo does **not** vendor it) |
| `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` | your Ethernet interface and the **active** IB rails |
| `MASTER_PORT` | any free port; must differ between concurrent jobs |

`MASTER_ADDR` is resolved automatically from the first node's IPv4 on `NCCL_SOCKET_IFNAME`
(compute hostnames on our cluster are IPv6 link-local only).

### Key hyperparameters (`configs/task/robodojo_flux2_klein_4b_actionpatch_full.yaml`)

```yaml
batch_size: 4                    # per GPU
gradient_accumulation_steps: 2   # -> global batch 4 x 2 x 32 = 256
learning_rate: 1e-4              # cosine schedule
num_epochs: 16                   # ~102.6k steps on the full RoboDojo set
save_every: 5000                 # ~5.5 h per checkpoint at 0.25 step/s
mixed_precision: bf16
weight_decay: 1e-2
num_workers: 16
```

`ZERO_STAGE=1` is exported by the launcher. Note that **ZeRO stage 2+ produced NaNs after
the first step** on this cluster; stage 1 and plain DDP are both fine.

### Key files

| Purpose | Path |
|---|---|
| AP model (action patching + dual loss) | `exploration/action_img_patch/model.py` |
| Model factory / variant dispatch | `exploration/action_img_patch/factory.py` |
| Model config | `configs/model/imagewam_flux2_klein_4b_actionpatch.yaml` |
| RoboDojo task config | `configs/task/robodojo_flux2_klein_4b_actionpatch_full.yaml` |
| RoboTwin task config | `configs/task/robotwin_flux2_klein_4b_actionpatch_full.yaml` |
| 4-node RoboDojo launcher | `exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh` |
| Train entry / accelerate | `scripts/train.py` · `scripts/flux2/train_flux2_klein_imagewam.sh` |
| HDF5 → LeRobot converter | `scripts/data/robodojo_raw_to_lerobot.py` |
| Dataset stats | `scripts/data/compute_robodojo_dataset_stats.py` |
| RoboDojo sim setup (Isaac Sim 5.1) | `scripts/setup_robodojo_sim.sh` · `scripts/install_robodojo_isaacsim_only.sh` |
| Eval managers | `experiments/robodojo/` |

### Operational notes

Hard-won on a 4-node H800 cluster over a 100k-step run; each of these cost real time.

**`weights/step_N.pt` and `state/step_N/` are different things.** The former is a single
~7.8 GB file for *evaluation*; the latter is a ~51 GB directory (sharded model + optimizer +
32 `random_states_*.pkl`) for *resuming*. A crash while writing can leave a complete
`weights/*.pt` next to a truncated `state/`. Before resuming, check that
`state/step_N/latest` exists and that the shard and `random_states` counts are full —
a truncated state directory will not resume.

**To tell whether a hung-looking job is actually hung, watch the log file size.** Over a
60 s window, a healthy run grows its Slurm `.out` by ~1.2 KB. Every other signal we tried
is ambiguous: GPU power sits at 130–210 W and `utilization.gpu` reads 100% *both* when
training normally and when spinning inside a stalled NCCL collective, and the main ranks'
`rchar` is small in both cases because the dataloader workers do the reading. Log growth
was the only reliable discriminator.

**Enable the NCCL FlightRecorder.** The launcher sets
`TORCH_NCCL_TRACE_BUFFER_SIZE=2048` and `TORCH_NCCL_DUMP_ON_TIMEOUT=1`, dumping to a shared
path rather than each node's `/tmp` (which vanishes with the job). Without it, a collective
timeout gives no stack, and the only way to find the culprit is the *absent-rank* method:
when all ranks but one report `Watchdog caught collective operation timeout` at the same
sequence number, the **one that reported nothing** is the rank that never reached the
collective — that is the faulty node. Note that torchrun's own "Root Cause" line names the
rank whose watchdog fired first, which is a *victim*, not the cause.

**Raising the collective timeout only covers PG 0.** `IMAGEWAM_PG_TIMEOUT_MIN` feeds
Accelerate's `InitProcessGroupKwargs`, which sets the default process group. The gradient
all-reduce runs on PG 1 and still uses the 10-minute NCCL default.

**Keep a node exclusion list.** One bad node can take down every 4-node job it lands on;
over this run a single host was implicated in five failures before being excluded. Note
that a command-line `--exclude` **overrides** the `#SBATCH -x` line rather than merging with
it, so either put the full list on the command line or omit it entirely.


## Basic Installation

ImageWAM uses `uv` to manage Python dependencies. Our recommended tested environment is CUDA 11.8, Python 3.11, and PyTorch 2.7.1.

```bash
uv sync --python 3.11 --extra shared
source .venv/bin/activate
```

Copy the local configuration template:

```bash
cp .env.example .env.local
```

Shell entrypoints under `scripts/` automatically read `.env.local`. You can write local paths there, or export them directly in your shell.

Common variables:

```bash
export DATA_ROOT=/path/to/datasets # Each dataset uses its own data root.
export MODEL_ROOT=/path/to/model/checkpoints # Used to store checkpoints.
export OUTPUT_ROOT=./runs
```

The basic installation only includes shared dependencies. Source paths, model weights, and extra dependencies for each model variant are described below.

## Model Preparation

This section prepares the required external model repositories and model downloads.

### FLUX.2 ImageWAM

We first describe the recommended FLUX.2 ImageWAM setup.

If you use a FLUX.2 variant, switch `transformers` to the FLUX-compatible version:

```bash
uv pip install "transformers==4.56.1"
```

Clone the FLUX.2 source code and check out the pinned commit:

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15

export FLUX2_SRC="$(pwd)/third_party/flux2"
```

Download FLUX.2 weights. Some FLUX.2 Hugging Face repositories may require access approval first.

```bash
# By default, this downloads FLUX.2 klein-base-4B, the autoencoder,
# and the 9B variant.
bash scripts/flux2/prepare_flux2_files.sh
```

For the default 4B variant, set:

```bash
export FLUX2_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="${MODEL_ROOT:-$(pwd)/checkpoints}/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
```

To use the 9B variant, set `FLUX2_VARIANT=9b` and point `FLUX2_MODEL_PATH` to the corresponding 9B weights.

### OmniGen2 ImageWAM

OmniGen2 ImageWAM is based on `VectorSpaceLab/OmniGen2@18e6f9d5271b517fcb32e999f10df943ae9b8f20`, with an additional patch required by ImageWAM.
The OmniGen2 variant uses `transformers==4.51.3`. If you previously switched versions for FLUX.2, switch back before running OmniGen2.

```bash
git clone https://github.com/yuyangalin/OmniGen2 third_party/OmniGen2

export OMNIGEN2_SRC="$(pwd)/third_party/OmniGen2"
export OMNIGEN2_MODEL_PATH=/path/to/OmniGen2/model
export QWEN_MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

### Ovis-U1 ImageWAM

This repository keeps Ovis-U1 code under `third_party/ovis_u1_hf`.

Ovis-U1 scripts use the Hugging Face model ID `AIDC-AI/Ovis-U1-3B` by default:

```bash
export OVIS_U1_MODEL_PATH=AIDC-AI/Ovis-U1-3B
```

You can also set `OVIS_U1_MODEL_PATH` to a local weights directory.

## Data Preparation

For LIBERO and RoboTwin, we use the preprocessed datasets provided by FastWAM.

### LIBERO

```bash
mkdir -p data/libero_mujoco3.3.2
huggingface-cli download yuanty/LIBERO-fastwam \
  --repo-type dataset \
  --local-dir data/libero_mujoco3.3.2
```

After downloading the archives, extract them:

```bash
cd data/libero_mujoco3.3.2
for f in *.tar.gz; do
  tar -xzf "$f"
done
cd ../..
```

Expected directory structure:

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

Set this when running LIBERO training or evaluation:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"
```

### RoboTwin

```bash
mkdir -p data/robotwin2.0
huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0
```

After downloading all split archives, concatenate and extract them:

```bash
cd data/robotwin2.0
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd ../..
```

Expected directory structure:

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

Set these when running RoboTwin training or evaluation:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"
```

To filter no-op frames in RoboTwin, we use a precomputed JSON file. It can be generated with:

```bash
bash scripts/data/precompute_noops_lerobot.sh
```

By default, this generates `${ROBOTWIN_ROOT}/nonidle_ranges.json`.

## Benchmark Environments

### LIBERO / LIBERO-plus

Benchmark environments are only required for evaluation.

```bash
# LIBERO
bash scripts/setup/_install_libero_env.sh

# LIBERO-plus
bash scripts/setup/_install_libero_plus_env.sh
```

Evaluation scripts use the following variable to activate the worker environment, because workers are launched in separate processes:

```bash
export LIBERO_WORKER_ENV_SOURCE=/path/to/imagewam/.venv/bin/activate
```

### RoboTwin

RoboTwin evaluation code is kept under `third_party/RoboTwin`, but assets and local simulator dependencies still need to be prepared on your machine.

```bash
bash scripts/setup/install_robotwin_env.sh
ln -sfn "$(pwd)/experiments/robotwin/imagewam_policy" "$(pwd)/third_party/RoboTwin/policy/imagewam_policy"
```

For asset preparation details, see `third_party/RoboTwin/README.vendor.md` and the upstream RoboTwin documentation.

## Training

Training wrappers automatically generate ActionDiT initialization weights if `ACTION_INIT` does not exist. To force regeneration, set `REBUILD_ACTION_INIT=true`.

### FLUX.2

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
FLUX2_VARIANT=4b \
PRECOMPUTE_QWEN3_CACHE=true \
bash scripts/flux2/run_train_flux2_klein_imagewam.sh
```

Common FLUX.2 overrides:

```bash
export FLUX2_VARIANT=4b          # 4b or 9b
export ZERO_STAGE=1              # 1/zero1 or 2/zero2
export QWEN_CACHE_DIR=/path/to/qwen3/cache # Optional; generated automatically if unset.
export ACTION_INIT=/path/to/action_dit_flux2_init.pt # Optional; generated automatically if unset.
```

### OmniGen2

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
PRECOMPUTE_QWEN_CACHE=true \
bash scripts/omnigen2/run_train_imagewam.sh
```

### Ovis-U1

LIBERO:

```bash
export DATA_ROOT="$(pwd)/data/libero_mujoco3.3.2"

GPU_PER_NODE=8 \
TASK_TYPE=libero \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

RoboTwin:

```bash
export DATA_ROOT="$(pwd)/data/robotwin2.0"
export ROBOTWIN_ROOT="${DATA_ROOT}/robotwin2.0"

GPU_PER_NODE=8 \
TASK_TYPE=robotwin \
bash scripts/ovis_u1/run_train_ovis_u1_imagewam.sh
```

## Evaluation

Evaluation scripts support directly specifying a checkpoint:

```bash
export CKPT_PATH=/path/to/model.pt
export DATASET_STATS_PATH=/path/to/dataset_stats.json
```

They also support deriving paths from a training run directory and step:

```bash
export EXP_PATH=/path/to/runs/{task}/{run_id}
export EVAL_TRAIN_STEP=10000
```

After setting `EXP_PATH`, the wrappers use:

```text
CKPT_PATH=${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt
DATASET_STATS_PATH=${EXP_PATH}/dataset_stats.json
```

Set the number of GPUs for evaluation with `NUM_GPUS`.

### FLUX.2

LIBERO:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_libero.sh
```

LIBERO-plus:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=9b \
bash scripts/flux2/run_eval_flux2_libero_plus.sh
```

RoboTwin:

```bash
NUM_GPUS=8 \
FLUX2_VARIANT=4b \
bash scripts/flux2/run_eval_flux2_robotwin.sh
```

RoboTwin evaluation enables `EVALUATION.skip_get_obs_within_replan=true` by default to speed up evaluation. If you need to save fully rendered videos, set `SKIP_GET_OBS_WITHIN_REPLAN=false`.

### OmniGen2

LIBERO:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero.sh
```

LIBERO-plus:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_libero_plus.sh
```

RoboTwin:

```bash
NUM_GPUS=8 bash scripts/omnigen2/run_eval_omnigen2_robotwin.sh
```

### Ovis-U1

LIBERO-plus:

```bash
NUM_GPUS=8 bash scripts/ovis_u1/run_eval_ovis_libero_plus.sh
```

## Release Checkpoints

The following FLUX.2 ImageWAM checkpoints are available on Hugging Face:

- `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`
- `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin`
- `yuyangalin/ImageWAM-FLUX.2-9B-LIBERO`

Checkpoints of other variants will be released later. Stay focused!

```bash
mkdir -p checkpoints/imagewam_release/libero/flux2_klein_4b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-4B-LIBERO \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/libero/flux2_klein_4b

mkdir -p checkpoints/imagewam_release/robotwin/flux2_klein_4b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/robotwin/flux2_klein_4b

mkdir -p checkpoints/imagewam_release/libero/flux2_klein_9b
huggingface-cli download yuyangalin/ImageWAM-FLUX.2-9B-LIBERO \
  --repo-type model \
  --local-dir checkpoints/imagewam_release/libero/flux2_klein_9b
```

Each model directory is expected to contain `model.pt`, `dataset_stats.json`, and the original training config, usually `train_config.yaml`.

Example: evaluate the released FLUX.2 LIBERO checkpoint:

```bash
export CKPT_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/model.pt"
export DATASET_STATS_PATH="$(pwd)/checkpoints/imagewam_release/libero/flux2_klein_4b/dataset_stats.json"

NUM_GPUS=8 FLUX2_VARIANT=4b bash scripts/flux2/run_eval_flux2_libero.sh
```

## Acknowledgements

ImageWAM is built on several codebases:

- Image-editing backbones built in or used by this repository: OmniGen2, FLUX.2, and Ovis-U1.
- This repository's code framework is based on FastWAM. We thank the authors for their excellent work.
- This codebase uses evaluation code from RoboTwin and LIBERO/LIBERO-plus.

## Citation

If you find this repository helpful for your research, please cite our paper:

```bibtex
@misc{zhang2026imagewam,
      title={ImageWAM: Do World Action Models Really Need Video Generation, or Just Image Editing?}, 
      author={Yuyang Zhang and Wenyao Zhang and Zekun Qi and He Zhang and Haitao Lin and Jingbo Zhang and Yao Mu and Xiaokang Yang and Wenjun Zeng and Xin Jin},
      year={2026},
      eprint={2606.19531},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.19531}, 
}
```

