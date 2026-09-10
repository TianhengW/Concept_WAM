#!/usr/bin/env bash
# action-router, 4 GPUs on gnho006. global batch = 4 x 4gpu x accum4 = 64
# (identical to the 1/20 AP line; only the sampler sharding differs from 8-GPU).
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnho006
#SBATCH --cpus-per-task=32
#SBATCH -J ap_router_1node
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
export MASTER_PORT="${MASTER_PORT:-29591}"
echo "[sbatch] action-router 4gpu@gnho006 | node=${SLURM_JOB_NODELIST}"

/soft/slurm/bin/srun --export=ALL --kill-on-bad-exit=1 bash -c "
  export NNODES=1 NODE_RANK=0
  export PYTHONPATH=/storage/yukaichengLab/mazijian/wth/ImageWAM:\${PYTHONPATH:-}
  export TASK=robotwin_flux2_klein_4b_actionrouter_sub20
  GPU_PER_NODE=4 bash scripts/flux2/train_flux2_klein_imagewam.sh 4 wandb.mode=offline gradient_accumulation_steps=4
"
