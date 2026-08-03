#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v2_armA_4node
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p v2/slurm_logs
source .venv/bin/activate

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export ZERO_STAGE=2
export TASK_BASENAME=robotwin_v2_armA

# Compute-node hostnames only resolve to IPv6 link-local; use the first node's
# eno1 IPv4 for rendezvous.
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29535}"
echo "[sbatch] nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# One RUN_ID shared across nodes (all ranks must write to the same run dir).
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

# 4 nodes x 8 GPUs x batch 1 x accum 4 = global 128. Conservative for the full
# Qwen fine-tune; bump batch_size once the first run confirms the memory headroom.
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export GPU_PER_NODE=8
  bash v2/scripts/launch_v2.sh \
    batch_size=4 \
    gradient_accumulation_steps=2 \
    max_steps=148000 \
    save_every=10000 \
    eval_every=100000 \
    num_epochs=8 data.train.episode_index_filter=null data.val.episode_index_filter=null log_every=10 \
    "$@"
' _ "$@"
