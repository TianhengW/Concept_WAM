#!/usr/bin/env bash
# TRAIN SMOKE: gnho020 x 7 GPUs, aug_full mixed data, 30 steps only.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH -w gnho020
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=56
#SBATCH -J aug_train_smoke
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK=robotwin_flux2_klein_4b_actionpatch_aug_full

GPU_PER_NODE=7 bash scripts/flux2/train_flux2_klein_imagewam.sh 7 \
  wandb.mode=offline max_steps=30 save_every=100000 \
  output_dir=./runs/train/aug_smoke_$SLURM_JOB_ID
