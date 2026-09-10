#!/usr/bin/env bash
# action-as-patch + FROZEN image-grounded Qwen3-VL text encoder, 4 nodes x 8 GPUs (FULL no-op-filtered data).
# 与 sbatch_actionpatch_full_4node.sh 超参逐项一致,唯一差异:文本编码器 Qwen3-4B -> 冻结 Qwen3-VL-4B(看 t0 帧)。
# 需 .venv_v3 (transformers 4.57.5, 提供 Qwen3VLForConditionalGeneration)。
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J ap_patchvl_full4n
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv_v3/bin/activate
export PATH="/soft/slurm/bin:${PATH}"

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"

FIRST_NODE=$(/soft/slurm/bin/scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29537}"
echo "[sbatch] action-as-patch-VL 4node | nodes=${SLURM_JOB_NODELIST} master=${MASTER_ADDR}:${MASTER_PORT}"

# 32 GPUs x batch 4 x accum 2 = global 256. Data: full no-op-filtered (~5.37M/epoch).
srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"
  export TASK=robotwin_flux2_klein_4b_actionpatch_vl_full
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    wandb.mode=offline "$@"
' _ "$@"
