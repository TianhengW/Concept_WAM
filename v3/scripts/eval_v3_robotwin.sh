#!/usr/bin/env bash
# RoboTwin evaluation for the v3 (VL LatentReasoner) checkpoint — wth-native.
# Borrows the simulator libs (curobo/sapien/warp) from the Automodel checkout, but
# from a CLONE of its venv upgraded to transformers 4.57.5 (Qwen3-VL) to match training.
set -euo pipefail

WROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
AROOT=/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM   # simulator libs only
FLUX2_SRC=/storage/yukaichengLab/mazijian/wth/flux2
SHARE=/storage/yukaichengLab/share/model
EVAL_VENV="${EVAL_VENV:-${AROOT}/.venv_v3eval}"                            # sim libs + transformers 4.57.5

EXP_PATH="${EXP_PATH:-${WROOT}/v3/runs/robotwin_v3/2026-07-31_13-35-24}"
EVAL_TRAIN_STEP="${EVAL_TRAIN_STEP:-002500}"
CKPT_PATH="${CKPT_PATH:-${EXP_PATH}/checkpoints/weights/step_${EVAL_TRAIN_STEP}.pt}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${WROOT}/data/robotwin2.0/dataset_stats.json}"

NUM_GPUS="${NUM_GPUS:-8}"
MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-1}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
PHASES="${PHASES:-[clean,random]}"
TASK_NAME="${TASK_NAME:-null}"

if [ ! -f "${CKPT_PATH}" ]; then echo "checkpoint not found: ${CKPT_PATH}" >&2; exit 1; fi

cd "${WROOT}"
source "${EVAL_VENV}/bin/activate"

export HF_HUB_OFFLINE=1
export PATH="${AROOT}/.local_bin:${PATH}"                       # static ffmpeg
export ROBOTWIN_EPISODE_TIMEOUT_S="${ROBOTWIN_EPISODE_TIMEOUT_S:-600}"
export ROBOTWIN_TASK_STALL_TIMEOUT_S="${ROBOTWIN_TASK_STALL_TIMEOUT_S:-1800}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_VENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
# wth code: imagewam -> wth/src (matches checkpoint), imagewam_v3 -> wth/v3, flux2 src.
export PYTHONPATH="${WROOT}/v3:${WROOT}/src:${FLUX2_SRC}/src:${FLUX2_SRC}"
export WORKER_PYTHONPATH="${PYTHONPATH}"

echo "[eval_v3] wth-native eval; ckpt=${CKPT_PATH}"
echo "[eval_v3] num_gpus=${NUM_GPUS} tasks/gpu=${MAX_TASKS_PER_GPU} episodes=${EVAL_NUM_EPISODES} phases=${PHASES} task_name=${TASK_NAME}"

if [ "${TASK_NAME}" = "null" ]; then
  TASK_ARG="EVALUATION.task_name=null"
else
  TASK_ARG="EVALUATION.task_name='${TASK_NAME}'"
fi

python experiments/robotwin/run_robotwin_manager.py \
  --config-name sim_robotwin \
  task=robotwin_flux2_klein_4b_base_imagewam \
  ckpt="${CKPT_PATH}" \
  EVALUATION.dataset_stats_path="${DATASET_STATS_PATH}" \
  EVALUATION.eval_num_episodes="${EVAL_NUM_EPISODES}" \
  "${TASK_ARG}" \
  EVALUATION.action_horizon=16 \
  EVALUATION.replan_steps=16 \
  model=imagewam_flux2_klein_4b_v3 \
  model.flux2_src_path="${FLUX2_SRC}" \
  model.flux2_model_path="${SHARE}/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors" \
  model.ae_model_path="${SHARE}/FLUX.2-dev/ae.safetensors" \
  model.vl_model_path="${SHARE}/Qwen3-VL-4B-Instruct" \
  model.variant=klein-base-4b \
  model.pack_proprio_after_text=true \
  model.proprio_dim=14 \
  MULTIRUN.enabled=true \
  MULTIRUN.num_gpus="${NUM_GPUS}" \
  MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU}" \
  "MULTIRUN.phases=${PHASES}" \
  "$@"
