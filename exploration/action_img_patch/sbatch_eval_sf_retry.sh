#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -J eval_sf_retry
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 6:00:00
# 用法: EXP_PATH=<selfflow run dir> UNITS_FILE=<每行 "task phase"> OUT_TAG=<输出子目录名> sbatch [-x ...] 本脚本
set -uo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"; mkdir -p "slurm_logs/${OUT_TAG}"
source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=600
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CKPT="${EXP_PATH}/checkpoints/weights/step_041940.pt"
STATS="${EXP_PATH}/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/${OUT_TAG}"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"
run_one() {
  local task=$1 phase=$2 gpu=$3
  local cfg="demo_clean"; [ "$phase" = "random" ] && cfg="demo_randomized"
  timeout -s KILL 5400 python experiments/robotwin/eval_robotwin_single.py \
    ckpt="${CKPT}" gpu_id=${gpu} EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
    EVALUATION.output_dir="${OUT}/${task}_${phase}" ${COMMON} \
    > "slurm_logs/${OUT_TAG}/${task}_${phase}.log" 2>&1
  echo "[done] ${task} ${phase} rc=$?"
}
i=0
while read -r task phase; do
  [ -z "${task}" ] && continue
  run_one "${task}" "${phase}" $((i % 8)) &
  i=$((i + 1))
  if [ $((i % 8)) -eq 0 ]; then wait; fi
done < "${UNITS_FILE}"
wait
echo "=== RETRY RESULTS ==="
for f in "slurm_logs/${OUT_TAG}"/*.log; do echo "--- $(basename $f) ---"; grep -E "Success rate:" "$f" | tail -1; done
echo "RETRY DONE"
