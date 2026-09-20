#!/usr/bin/env bash
# action-as-patch x RoboCasa365 PRETRAIN (Human300), 2 nodes x 8 GPUs, H800/H100 cluster.
# Modeled on scripts/sbatch_actionpatch_2node.sh (proven eno1 rendezvous + ZeRO-1 on this cluster).
# Usage:  BATCH=<per-gpu batch from smoke> sbatch experiments/robocasa/sbatch_robocasa_pretrain_2node.sh [hydra overrides...]
#SBATCH -p yukaichenglab
#SBATCH -N 2
# ^ pin to the two idle H100-80GB nodes (same node type as the smoke); override with `sbatch -w <nodes>`
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH -J ap_rc365_pre2n
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
export PATH=/soft/slurm/bin:${PATH}   # scontrol/srun are not on PATH in non-interactive shells on this cluster

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source /storage/yukaichengLab/mazijian/wth/ImageWAM/.venv/bin/activate

BATCH="${BATCH:?set BATCH=<per-gpu batch_size> (from the 007 smoke sweep)}"
# Rendezvous NIC = the 192.168.x/16 Ethernet fabric (eno1 on gnho H800 nodes, enp86s0f0np0 on nv-h100 nodes).
NET_IF="${NET_IF:-$(ip -o -4 addr show | awk '$4 ~ /^192\.168\./ {print $2; exit}')}"

export NCCL_SOCKET_IFNAME="${NET_IF}"
export GLOO_SOCKET_IFNAME="${NET_IF}"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1          # ZeRO-1: optimizer states sharded across 16 ranks (RoboTwin 2-node recipe on this cluster)
export FLUX2_SRC="${REPO_ROOT}"
export RUN_ID_SYNC_TIMEOUT=600

FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show ${NET_IF} | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29560}"
echo "[sbatch] AP RoboCasa365 pretrain 2node | nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT} batch/gpu=${BATCH} global=$((BATCH*16)) if=${NET_IF}"

srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export TASK=robocasa_flux2_klein_4b_actionpatch_pretrain
  bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    batch_size='"${BATCH}"' \
    wandb.mode=offline "$@"
' _ "$@"
