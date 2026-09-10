#!/usr/bin/env bash
# TRAIN SMOKE for action-as-patch on RoboDojo: 1 node x 8 GPUs, 30 optimizer steps.
# Usage: sbatch exploration/action_img_patch/sbatch_robodojo_train_smoke.sh [extra hydra overrides]
# Env: SMOKE_DATASET_DIR (default: the partial smoke conversion), SMOKE_NONIDLE (json or "null"),
#      SMOKE_STATS (dataset_stats.json or "null" to compute on the fly).
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J ap_robodojo_smoke2g
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
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TASK=robodojo_flux2_klein_4b_actionpatch_full

DS="${SMOKE_DATASET_DIR:-/storage/yukaichengLab/share/datasets/RoboDojo_lerobot_smoke}"
NONIDLE="${SMOKE_NONIDLE:-${DS}/nonidle_ranges.json}"
STATS="${SMOKE_STATS:-${DS}/dataset_stats.json}"

echo "[smoke] dataset=${DS} nonidle=${NONIDLE} stats=${STATS} $(date)"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8

GPU_PER_NODE=2 bash scripts/flux2/train_flux2_klein_imagewam.sh 2 \
  wandb.mode=offline max_steps=10 save_every=5 log_every=1 batch_size=2 num_workers=4 \
  "data.train.dataset_dirs=[${DS}]" "data.val.dataset_dirs=[${DS}]" \
  "data.train.nonidle_filter_path=${NONIDLE}" "data.val.nonidle_filter_path=${NONIDLE}" \
  "data.train.pretrained_norm_stats=${STATS}" "data.val.pretrained_norm_stats=${STATS}" \
  output_dir=./runs/train/robodojo_smoke_${SLURM_JOB_ID} "$@"
echo "[smoke] TRAIN_SMOKE_DONE status=$? $(date)"
