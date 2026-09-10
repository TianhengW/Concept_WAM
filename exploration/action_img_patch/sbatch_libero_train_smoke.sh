#!/usr/bin/env bash
# LIBERO / LIBERO-plus action-as-patch TRAIN SMOKE: 1 node x 8 GPUs, 30 steps.
# Verifies data loading (camera keys, 224x448 concat), 7-d codec, loss finite, and per-GPU memory
# at the requested batch size (a background nvidia-smi sampler writes slurm_logs/<job>_<id>.gpumem).
#   TASK=libero_flux2_klein_4b_actionpatch | libero_plus_flux2_klein_4b_actionpatch
#   BS=<per-GPU batch>   (default 8)
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J libero_ap_smoke
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
export ZERO_STAGE="${ZERO_STAGE:-1}"
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK="${TASK:-libero_flux2_klein_4b_actionpatch}"
BS="${BS:-8}"
MAX_STEPS="${MAX_STEPS:-30}"
echo "[smoke] task=${TASK} bs=${BS} zero=${ZERO_STAGE} node=${SLURMD_NODENAME} max_steps=${MAX_STEPS}"

# GPU memory sampler (peak per GPU is what we want).
GPUMEM_LOG="slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.gpumem"
( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | \
    awk -v t="$(date +%H:%M:%S)" '{print t","$0}' >> "${GPUMEM_LOG}"; sleep 15; done ) &
SAMPLER_PID=$!
trap 'kill ${SAMPLER_PID} 2>/dev/null || true' EXIT

GPU_PER_NODE=${NGPU:-8} bash scripts/flux2/train_flux2_klein_imagewam.sh ${NGPU:-8} \
  wandb.mode=offline max_steps="${MAX_STEPS}" save_every=100000 batch_size="${BS}" \
  output_dir="./runs/train/${TASK}_smoke_${SLURM_JOB_ID}"

echo "[smoke] finished; peak GPU mem (MiB) per index:"
awk -F, '{ if ($3+0 > m[$2]) m[$2]=$3+0 } END { for (i in m) print i, m[i] }' "${GPUMEM_LOG}" | sort -n
