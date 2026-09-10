#!/usr/bin/env bash
# Eval ap_patch (single-stream action-as-patch, 1/20-data run) final ckpt step_041940.
# 50 tasks x {clean,random} x 10 episodes, 1 node x 8 GPUs.
# Runs wth/ImageWAM code with the eval-ready venv + render patch from the
# Automodel checkout (wth/.venv lacks sim deps).
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J eval_ap_patch41940
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source "${EVAL_ENV_ROOT}/.venv/bin/activate"

export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S="${ROBOTWIN_EPISODE_TIMEOUT_S:-600}"
export ROBOTWIN_TASK_STALL_TIMEOUT_S="${ROBOTWIN_TASK_STALL_TIMEOUT_S:-1800}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# exploration package + wth sources take precedence over the venv checkout
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export FLUX2_MODEL_PATH="/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC="/storage/yukaichengLab/share/model/Qwen3-4B"

export TASK="robotwin_flux2_klein_4b_actionpatch_sub20"
export EXP_PATH="${EXP_PATH:-${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_sub20/2026-08-07_11-04-27}"
export EVAL_TRAIN_STEP="${EVAL_TRAIN_STEP:-041940}"
export EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
export NUM_GPUS=8

bash scripts/flux2/run_eval_flux2_robotwin.sh "$@"
