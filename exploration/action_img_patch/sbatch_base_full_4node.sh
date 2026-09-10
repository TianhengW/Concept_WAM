#!/usr/bin/env bash
# BASELINE for exploration/action_img_patch: original ImageWAM (MoT), 4 nodes x 8 GPUs (FULL no-op-filtered data).
# Hyper-parameters identical to sbatch_actionpatch_2node.sh (only the model differs).
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J ap_base_full4n
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"

# eno1 IPv4 of the first node for rendezvous (compute hostnames are IPv6 link-local only).
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29533}"
echo "[sbatch] actionpatch-BASELINE (original MoT) 2node | nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# 32 GPUs x batch 4 x accum 2 = global 256. Data: full no-op-filtered (~5.37M/epoch).
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"
  export TASK=robotwin_flux2_klein_4b_base_full
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    wandb.mode=offline "$@"
' _ "$@"
