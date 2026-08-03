#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007
#SBATCH -J ts_latent_4node
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
export ROBOTWIN_ROOT="/storage/yukaichengLab/share/datasets/fastwam_robotwin/robotwin2.0"
export DATA_ROOT="/storage/yukaichengLab/share/datasets/fastwam_robotwin"

# eno1 IPv4 of the first node for rendezvous (compute hostnames are IPv6 link-local only).
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29513}"
echo "[sbatch] three-stream+latent | nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# 32 GPUs x batch 4 x accum 2 = global 256. Data setting matches ImageWAM release
# (nonidle_ranges.json baked into the task config). 10 epochs.
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export TASK=robotwin_flux2_klein_4b_threestream_latent_imagewam
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    num_epochs=10 \
    batch_size=4 \
    gradient_accumulation_steps=2 \
    log_every=10 \
    save_every=2500 \
    eval_every=100000 \
    wandb.mode=offline "$@"
' _ "$@"
