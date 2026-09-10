#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1 FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29599 NNODES=1 NODE_RANK=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK=robotwin_flux2_klein_4b_actionrouter_sub20
GPU_PER_NODE=2 bash scripts/flux2/train_flux2_klein_imagewam.sh 2 \
  wandb.mode=offline batch_size=2 max_steps=10 save_every=5 log_every=1 num_workers=4
