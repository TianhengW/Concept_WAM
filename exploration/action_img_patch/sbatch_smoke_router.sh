#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -x nv-h100-029,nv-h100-007,gnho020,gnho009,gnho017,gnho036
#SBATCH -J smoke_router_fm
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source "${REPO_ROOT}/.venv/bin/activate"
export NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1 FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29600 NNODES=1 NODE_RANK=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK=robotwin_flux2_klein_4b_actionrouter_sub20
GPU_PER_NODE=2 bash scripts/flux2/train_flux2_klein_imagewam.sh 2 \
  wandb.mode=offline batch_size=2 max_steps=12 save_every=6 log_every=1 num_workers=4
