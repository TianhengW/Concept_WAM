#!/bin/bash
# 100ep dual-phase eval of aug-mixed run ckpt step_112000 (50 tasks x clean+random x 100ep).
# Cloned from sbatch_eval_100ep_2node.sh; only CKPT/STATS/dirs changed. seed=43 same as baseline.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -J eval_100ep_aug112000
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 14:00:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"

export SEED="${SEED:-43}"
export REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
export EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"
export LOGDIR="slurm_logs/eval_100ep_aug112000_seed${SEED}"
export RESDIR="${REPO_ROOT}/evaluate_results/robotwin/eval_100ep_aug112000_seed${SEED}"
mkdir -p "${LOGDIR}" "${RESDIR}"

D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_aug_full/2026-08-26_08-15-54"
export CKPT="${D}/checkpoints/weights/step_112000.pt"
STATS="${D}/dataset_stats.json"
export COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=100 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"

TASKS="adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size click_alarmclock click_bell dump_bin_bigbin grab_roller handover_block handover_mic hanging_mug lift_pot move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad open_laptop open_microwave pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right place_bread_basket place_bread_skillet place_burger_fries place_can_basket place_cans_plasticbox place_container_plate place_dual_shoes place_empty_cup place_fan place_mouse_pad place_object_basket place_object_scale place_object_stand place_phone_stand place_shoe press_stapler put_bottles_dustbin put_object_cabinet rotate_qrcode scan_object shake_bottle shake_bottle_horizontally stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two stamp_seal turn_switch"
GLIST=""
for t in ${TASKS}; do GLIST="${GLIST} ${t}:demo_clean ${t}:demo_randomized"; done
export GLIST

echo "[launcher] aug112000 seed=${SEED} groups=$(echo ${GLIST} | wc -w) nodes=${SLURM_JOB_NODELIST}"
srun --ntasks=1 --ntasks-per-node=1 --gres=gpu:8 --export=ALL bash exploration/action_img_patch/eval_100ep_worker_1n.sh
echo "[launcher] aug112000 seed=${SEED} COMPLETE"
