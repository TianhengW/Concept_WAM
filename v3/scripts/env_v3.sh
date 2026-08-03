#!/usr/bin/env bash
# Shared environment for v3 (VL LatentReasoner) training. Source from launch scripts.
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/storage/yukaichengLab/mazijian/wth/ImageWAM}"
export V3_ROOT="${REPO_ROOT}/v3"

export FLUX2_SRC="${FLUX2_SRC:-/storage/yukaichengLab/mazijian/wth/flux2}"
export FLUX2_MODEL_PATH="${FLUX2_MODEL_PATH:-/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors}"
export FLUX2_AE_MODEL_PATH="${FLUX2_AE_MODEL_PATH:-/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors}"
export VL_MODEL_PATH="${VL_MODEL_PATH:-/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct}"
export DATA_DIR="${DATA_DIR:-/storage/yukaichengLab/share/datasets/fastwam_robotwin/robotwin2.0}"
export BASE_CKPT="${BASE_CKPT:-/storage/yukaichengLab/share/model/ImageWAM/imagewam_release/robotwin/flux2_klein_4b/model.pt}"

# imagewam_v3 importable, plus the base package and the flux2 source.
export PYTHONPATH="${V3_ROOT}:${REPO_ROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
