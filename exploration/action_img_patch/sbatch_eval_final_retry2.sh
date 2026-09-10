#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=gnho031
#SBATCH -J eval_final_retry2
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 1:10:00
set -uo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"; mkdir -p slurm_logs/retry_final
source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-11_14-38-46"
CKPT="${D}/checkpoints/weights/step_104850.pt"
STATS="runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-11_14-38-46/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/retry_final"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"
run_one() {
  local task=$1 cfg=$2 gpu=$3
  timeout -s KILL 2400 python experiments/robotwin/eval_robotwin_single.py \
    ckpt="${CKPT}" gpu_id=${gpu} EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
    EVALUATION.output_dir="${OUT}/${task}_${cfg}" ${COMMON} \
    > "slurm_logs/retry_final/${task}_${cfg}.log" 2>&1
  echo "[done] ${task} ${cfg} rc=$?"
}
run_one move_stapler_pad demo_randomized 0 &
run_one place_mouse_pad   demo_clean      1 &
wait
echo "=== FINAL RETRY 2-GROUP RESULTS ==="
for f in slurm_logs/retry_final/*.log; do echo "--- $(basename $f) ---"; grep -E "Success rate:" "$f" | tail -1; done
echo "FINAL RETRY DONE"
