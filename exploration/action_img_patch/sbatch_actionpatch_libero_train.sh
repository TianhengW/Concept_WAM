#!/usr/bin/env bash
# action-as-patch on LIBERO / LIBERO-plus, multi-node (H800). Submit with the node count:
#   sbatch -N 2 -J ap_libero_2n   TASK=... exploration/action_img_patch/sbatch_actionpatch_libero_train.sh
# Env knobs (all optional):
#   TASK    = libero_flux2_klein_4b_actionpatch | libero_plus_flux2_klein_4b_actionpatch
#   BS      = per-GPU batch (default 8)      ACCUM = grad accumulation (default 1)
#   EPOCHS  = num_epochs (default from task yaml)   RESUME = path to trainer state dir (optional)
# global batch = BS * ACCUM * 8 * SLURM_NNODES. Hyper-parameters otherwise identical to the
# RoboTwin action-as-patch recipe (lr 1e-4 cosine, wd 1e-2, bf16, ZeRO-1).
#SBATCH -p yukaichenglab
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J ap_libero
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
export ZERO_STAGE="${ZERO_STAGE:-1}"
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export TASK="${TASK:-libero_flux2_klein_4b_actionpatch}"
BS="${BS:-8}"
ACCUM="${ACCUM:-1}"
EXTRA=()
[ -n "${EPOCHS:-}" ] && EXTRA+=("num_epochs=${EPOCHS}")
[ -n "${RESUME:-}" ] && EXTRA+=("resume=${RESUME}")

FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29563}"
GLOBAL=$((BS * ACCUM * 8 * SLURM_NNODES))
echo "[sbatch] ${TASK} | nodes=${SLURM_JOB_NODELIST} (${SLURM_NNODES}) master=${MASTER_ADDR}:${MASTER_PORT} | bs=${BS} accum=${ACCUM} global=${GLOBAL} extra=${EXTRA[*]:-}"

srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    wandb.mode=offline "$@"
' _ batch_size="${BS}" gradient_accumulation_steps="${ACCUM}" "${EXTRA[@]}" "$@"
