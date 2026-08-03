#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v3_smoke
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v3/slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p v3/slurm_logs
# v3 needs transformers>=4.57 (Qwen3-VL): use the isolated .venv_v3.
source .venv_v3/bin/activate

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export MASTER_PORT=29544

export NNODES=1 NODE_RANK=0 GPU_PER_NODE=8 ZERO_STAGE=2
export TASK_BASENAME=robotwin_v3_smoke

# Small, fast smoke: a handful of steps, one save to exercise the checkpoint path.
bash v3/scripts/launch_v3.sh \
  batch_size=1 \
  gradient_accumulation_steps=1 \
  max_steps=6 \
  num_epochs=1 \
  save_every=5 \
  eval_every=100000 \
  log_every=1 \
  num_workers=4 \
  prefetch_factor=2 \
  "$@"
