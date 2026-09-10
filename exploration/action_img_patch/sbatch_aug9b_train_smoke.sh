#!/usr/bin/env bash
# 9B TRAIN SMOKE: 1 node x 8 GPUs, aug_full mixed data, 30 steps. 验证9B+Qwen3-8B显存.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho008,gnho016,gnho036,gnho038
#SBATCH -J aug9b_smoke
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate
export PATH="/soft/slurm/bin:${PATH}"
export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK=robotwin_flux2_klein_9b_actionpatch_aug_full
GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
  wandb.mode=offline max_steps=30 save_every=100000 \
  output_dir=./runs/train/aug9b_smoke_$SLURM_JOB_ID
