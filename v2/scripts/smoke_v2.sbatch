#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -J v2_armA_smoke
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p v2/slurm_logs
source .venv/bin/activate

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export MASTER_PORT=29534

# Single node: no cross-node rendezvous needed.
export NNODES=1 NODE_RANK=0 GPU_PER_NODE=8 ZERO_STAGE=2
export TASK_BASENAME=robotwin_v2_armA_smoke

# Small, fast smoke: a handful of steps, one save to exercise the checkpoint path.
bash v2/scripts/launch_v2.sh \
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
