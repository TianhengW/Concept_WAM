#!/usr/bin/env bash
# action-as-patch, 4 nodes x 8 GPUs: FULL no-op-filtered data + Robotwin_aug
# hard-task augmentation. Hyper-parameters identical to
# sbatch_actionpatch_full_4node.sh (only the data mix differs).
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J ap_patch_aug9b4n
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate
export PATH="/soft/slurm/bin:${PATH}"

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"

FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29537}"
echo "[sbatch] action-as-patch aug 9B 4node | nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# 32 GPUs x batch 2 x accum 4 = global 256 (9B). Data: full no-op-filtered + aug hard tasks.
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"
  export TASK=robotwin_flux2_klein_9b_actionpatch_aug_full
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    wandb.mode=offline "$@"
' _ "$@"
