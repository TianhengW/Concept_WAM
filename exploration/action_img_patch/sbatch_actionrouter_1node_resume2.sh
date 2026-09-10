#!/usr/bin/env bash
# action-router (flow-matching head + anti-collapse), RESUME from step_020000
# after job 93122 died at ~step 22030 (node gnho011, external kill, no traceback).
# Same hyper-parameters as the 1/20 AP line; global batch 64, lr 1e-4 cosine, 10ep.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho020,gnho009,gnho017,gnho036,gnho011,gnho032
#SBATCH -J ap_router_resume2
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source "${REPO_ROOT}/.venv/bin/activate"

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29594}"
RESUME_DIR="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionrouter_sub20/2026-08-25_17-00-18/checkpoints/state/step_030000"
echo "[sbatch] action-router RESUME | node=${SLURM_JOB_NODELIST} | resume=${RESUME_DIR}"

/soft/slurm/bin/srun --export=ALL --kill-on-bad-exit=1 bash -c "
  export NNODES=1 NODE_RANK=0
  export PYTHONPATH=/storage/yukaichengLab/mazijian/wth/ImageWAM:\${PYTHONPATH:-}
  export TASK=robotwin_flux2_klein_4b_actionrouter_sub20
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 wandb.mode=offline resume=${RESUME_DIR}
"
