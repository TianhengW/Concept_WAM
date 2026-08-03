#!/usr/bin/env bash
# Shared environment for Arm A (v2) training. Source this from launch scripts.
# All paths are taken from the working FLUX.2-4B RoboTwin setup on this cluster.
set -euo pipefail

# Repo root = ImageWAM (the parent of v2/).
export REPO_ROOT="${REPO_ROOT:-/storage/yukaichengLab/mazijian/wth/ImageWAM}"
export V2_ROOT="${REPO_ROOT}/v2"

# Model / data paths (share storage).
export FLUX2_SRC="${FLUX2_SRC:-/storage/yukaichengLab/mazijian/wth/flux2}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors}"
export QWEN3_MODEL_SPEC="${QWEN3_MODEL_SPEC:-/storage/yukaichengLab/share/model/Qwen3-4B}"
export DATA_DIR="${DATA_DIR:-/storage/yukaichengLab/share/datasets/fastwam_robotwin/robotwin2.0}"
export BASE_CKPT="${BASE_CKPT:-/storage/yukaichengLab/share/model/ImageWAM/imagewam_release/robotwin/flux2_klein_4b/model.pt}"

# imagewam_v2 must be importable, plus the base package and the flux2 source.
export PYTHONPATH="${V2_ROOT}:${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
