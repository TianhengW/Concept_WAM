#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041,gnho009,gnho017
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 00:50:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
VENV="${EVAL_ENV_ROOT}/.venv_vleval"
cd "${REPO_ROOT}"
export PATH="${VENV}/bin:${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export HF_HUB_OFFLINE=1
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${VENV}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
PY="${VENV}/bin/python"
STEP="${STEP:?}"; GROUPS="${GROUPS:?}"
D="${RUNDIR:-${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_vl_full/2026-08-13_19-04-40}"
CKPT="${D}/checkpoints/weights/step_${STEP}.pt"
STATS="${D}/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/vl_${STEP}_10ep"
LOGDIR="slurm_logs/vl_${STEP}"; mkdir -p "$LOGDIR" "$OUT"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_vl_full EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.vl_model_path=/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"
is_done(){ local l=$1; [ -f "$l" ] && sed "s/\x1b\[[0-9;]*m//g" "$l" | grep -qE "Success rate: [0-9]+/10"; }
for grp in ${GROUPS}; do
  task=${grp%%:*}; cfg=${grp##*:}; LOG="${LOGDIR}/${task}_${cfg}.log"
  if is_done "$LOG"; then echo "SKIP $task $cfg"; continue; fi
  echo "start $task $cfg on $(hostname)"
  timeout -s KILL 2000 "$PY" experiments/robotwin/eval_robotwin_single.py \
    ckpt="${CKPT}" gpu_id=0 seed=42 EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
    EVALUATION.output_dir="${OUT}/${task}_${cfg}" ${COMMON} > "${LOG}" 2>&1 && echo "OK $task $cfg" || echo "FAIL $task $cfg"
done
echo "POISON DONE step=${STEP}"
