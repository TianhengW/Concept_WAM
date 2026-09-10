#!/usr/bin/env bash
# action-as-patch scheme A (time-aligned action RoPE), 1 full node x 8 GPUs.
# Hyper-parameters identical to the 1/20 AP line (SR 88.0 @ 41940): effective
# world_size=8, global batch = 4 x 8 x accum 2 = 64, lr 1e-4 cosine, 10 epochs.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho020,gnho009,gnho017,gnho036,gnho011,gnho032
#SBATCH -J ap_timealign_1node
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
export MASTER_PORT="${MASTER_PORT:-29601}"
EXTRA_ARGS="${1:-}"
echo "[sbatch] ap timealign 1node | node=${SLURM_JOB_NODELIST} | extra=${EXTRA_ARGS}"

/soft/slurm/bin/srun --export=ALL --kill-on-bad-exit=1 bash -c "
  export NNODES=1 NODE_RANK=0
  export PYTHONPATH=/storage/yukaichengLab/mazijian/wth/ImageWAM:\${PYTHONPATH:-}
  export TASK=robotwin_flux2_klein_4b_actionpatch_timealign_sub20
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 wandb.mode=offline ${EXTRA_ARGS}
"
