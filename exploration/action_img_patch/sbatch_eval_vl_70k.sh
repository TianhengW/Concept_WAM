#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041,gnho009,gnho017
#SBATCH -J eval_vl_70k
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 4:00:00
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

D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_vl_full/2026-08-13_19-04-40"
CKPT="${D}/checkpoints/weights/step_070000.pt"
STATS="${D}/dataset_stats.json"
OUT="${REPO_ROOT}/evaluate_results/robotwin/vl_70k_10ep"
LOGDIR="slurm_logs/vl_70k"; mkdir -p "$LOGDIR" "$OUT"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_vl_full EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.vl_model_path=/storage/yukaichengLab/share/model/Qwen3-VL-4B-Instruct model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"

TASKS="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch"
GLIST=""
for t in ${TASKS}; do GLIST="${GLIST} ${t}:demo_clean ${t}:demo_randomized"; done

echo "[vl_70k] groups=$(echo $GLIST|wc -w) ckpt=step_070000 venv=${VENV}"
i=0
for grp in ${GLIST}; do
  gpu=$((i % 8)); task=${grp%%:*}; cfg=${grp##*:}
  echo "[gpu$gpu] start $task $cfg"
  ( timeout -s KILL 3000 "$PY" experiments/robotwin/eval_robotwin_single.py \
      ckpt="${CKPT}" gpu_id=${gpu} seed=42 EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
      EVALUATION.output_dir="${OUT}/${task}_${cfg}" ${COMMON} \
      > "${LOGDIR}/${task}_${cfg}.log" 2>&1 || echo "[gpu$gpu] FAIL $task $cfg" ) &
  i=$((i+1))
  [ $((i % 8)) -eq 0 ] && wait
done
wait
echo "=== VL 70k 10ep DONE ==="
