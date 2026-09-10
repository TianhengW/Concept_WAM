#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041,gnho035,gnho036,gnho034,gnho018,gnho033
#SBATCH -J eval_final100ep_r26
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 14:00:00
set -uo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"; mkdir -p slurm_logs/retry_100ep
source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-11_14-38-46"
CKPT="${D}/checkpoints/weights/step_104850.pt"
STATS="runs/robotwin_flux2_klein_4b_actionpatch_full/2026-08-11_14-38-46/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/retry_100ep"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=100 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"
GLIST="blocks_ranking_size:demo_randomized hanging_mug:demo_clean hanging_mug:demo_randomized move_pillbottle_pad:demo_randomized place_can_basket:demo_clean place_can_basket:demo_randomized place_cans_plasticbox:demo_randomized place_object_basket:demo_randomized place_object_scale:demo_clean put_bottles_dustbin:demo_clean put_bottles_dustbin:demo_randomized put_object_cabinet:demo_clean put_object_cabinet:demo_randomized scan_object:demo_clean scan_object:demo_randomized shake_bottle:demo_randomized shake_bottle_horizontally:demo_randomized stack_blocks_three:demo_clean stack_blocks_three:demo_randomized stack_blocks_two:demo_randomized stack_bowls_three:demo_clean stack_bowls_three:demo_randomized stack_bowls_two:demo_clean stack_bowls_two:demo_randomized stamp_seal:demo_randomized turn_switch:demo_randomized"
echo "DEBUG 组数=$(echo $GLIST | wc -w)"
i=0
for grp in $GLIST; do
  gpu=$((i % 8)); task=${grp%%:*}; cfg=${grp##*:}
  echo "[gpu$gpu] start $task $cfg"
  ( timeout -s KILL 9000 python experiments/robotwin/eval_robotwin_single.py \
      ckpt="${CKPT}" gpu_id=${gpu} EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
      EVALUATION.output_dir="${OUT}/${task}_${cfg}" ${COMMON} \
      > "slurm_logs/retry_100ep/${task}_${cfg}.log" 2>&1 || echo "[gpu$gpu] FAIL $task $cfg" ) &
  i=$((i+1))
  [ $((i % 8)) -eq 0 ] && wait
done
wait
echo "=== ALL ${i} GLIST DONE ==="
