#!/usr/bin/env bash
# 2-GPU 10-step smoke for the A/B study. Usage (as sbatch script):
#   sbatch --gres=gpu:2 ... smoke_ab.sh <timealign|sigmashift|sigmashift_r1|ap_ref> <port>
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -x nv-h100-029,nv-h100-007,gnho020,gnho009,gnho017,gnho036,gnho011,gnho032
#SBATCH -J ap_ab_smoke
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
VARIANT="${1:?variant required: timealign|sigmashift|sigmashift_r1|ap_ref}"
PORT="${2:-29603}"

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source "${REPO_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1 FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${PORT}" NNODES=1 NODE_RANK=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

EXTRA=""
case "${VARIANT}" in
  timealign)     export TASK=robotwin_flux2_klein_4b_actionpatch_timealign_sub20
                 EXTRA="wandb.name=smoke_timealign" ;;
  sigmashift)    export TASK=robotwin_flux2_klein_4b_actionpatch_sigmashift_sub20
                 EXTRA="wandb.name=smoke_sigmashift" ;;
  sigmashift_r1) export TASK=robotwin_flux2_klein_4b_actionpatch_sigmashift_sub20
                 EXTRA="model.action_shift_ratio=1.0 wandb.name=smoke_sigmashift_r1" ;;
  ap_ref)        export TASK=robotwin_flux2_klein_4b_actionpatch_sub20
                 EXTRA="wandb.name=smoke_ap_ref" ;;
  *) echo "unknown variant ${VARIANT}"; exit 1 ;;
esac

echo "[smoke] variant=${VARIANT} task=${TASK} port=${PORT} extra=${EXTRA}"
GPU_PER_NODE=2 bash scripts/flux2/train_flux2_klein_imagewam.sh 2 \
  wandb.mode=offline batch_size=2 max_steps=10 save_every=5 log_every=1 num_workers=4 ${EXTRA}
