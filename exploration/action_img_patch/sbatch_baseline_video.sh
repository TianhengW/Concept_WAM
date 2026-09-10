#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041,gnho009,gnho017
#SBATCH -J base_video
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 3:00:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"; source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CKPT="/storage/yukaichengLab/share/model/ImageWAM-FLUX.2-4B-RoboTwin/model.pt"
STATS="/storage/yukaichengLab/share/model/ImageWAM-FLUX.2-4B-RoboTwin/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/baseline_mot_video"
NEPS="${NEPS:-20}"
CFG="${CFG:-demo_randomized}"
LOGDIR="slurm_logs/baseline_video"; mkdir -p "$LOGDIR" "$OUT"
COMMON="task=robotwin_flux2_klein_4b_base_imagewam EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=${NEPS} EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"

TASKS="${TASKS:-handover_block put_object_cabinet place_object_basket pick_diverse_bottles hanging_mug place_can_basket}"
i=0
for task in ${TASKS}; do
  gpu=$(( i % 8 ))
  echo "[gpu$gpu] start $task $CFG (${NEPS}ep, seed=42)"
  ( timeout -s KILL 7200 python experiments/robotwin/eval_robotwin_single.py \
      ckpt="${CKPT}" gpu_id=${gpu} seed=42 EVALUATION.task_name=${task} EVALUATION.task_config=${CFG} \
      EVALUATION.output_dir="${OUT}/${task}_${CFG}" ${COMMON} \
      > "${LOGDIR}/${task}_${CFG}.log" 2>&1 || echo "[gpu$gpu] FAIL $task" ) &
  i=$(( i + 1 ))
done
wait
echo "=== BASELINE VIDEO DONE ==="
