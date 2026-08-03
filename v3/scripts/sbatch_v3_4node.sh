#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v3_4node
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
# Full-param 4B VL LLM (+ coconut activations) on top of the base model is heavier
# than v2 Arm A; start at ZeRO-2 and bump to 3 if the smoke reports OOM.
export ZERO_STAGE="${ZERO_STAGE:-2}"
export TASK_BASENAME=robotwin_v3

# Compute-node hostnames only resolve to IPv6 link-local; use the first node's
# eno1 IPv4 for rendezvous.
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29545}"
echo "[sbatch] nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# One RUN_ID shared across nodes (all ranks write to the same run dir).
export RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

# 4 nodes x 8 GPUs (32) x batch 4 x accum 2 = global 256. Downsampled dataset
# (default periodic_prefix filter from the config, ~9% episodes), 5 epochs.
# NOTE: batch 4 with the 16-step coconut loop is memory-heavy (per-sample backward
# graph over all 16 steps). If this OOMs, submit with ZERO_STAGE=3.
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export GPU_PER_NODE=8
  bash v3/scripts/launch_v3.sh \
    batch_size=4 \
    gradient_accumulation_steps=2 \
    max_steps=40000 \
    save_every=2500 \
    eval_every=100000 \
    num_epochs=5 log_every=10 \
    "$@"
' _ "$@"
