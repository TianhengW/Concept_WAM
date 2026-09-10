#!/bin/bash
# 100ep dual-phase eval of aug-mixed run ckpt step_110000 (50 tasks x clean+random x 100ep).
# Cloned from sbatch_eval_100ep_2node.sh; only CKPT/STATS/dirs changed. seed=43 same as baseline.
#SBATCH -p yukaichenglab
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -J eval_100ep_aug9b110k_retry
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 14:00:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"

export SEED="${SEED:-43}"
export REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
export EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"
export LOGDIR="slurm_logs/eval_100ep_aug9b110k_seed${SEED}"
export RESDIR="${REPO_ROOT}/evaluate_results/robotwin/eval_100ep_aug9b110k_seed${SEED}"
mkdir -p "${LOGDIR}" "${RESDIR}"

D="${REPO_ROOT}/runs/robotwin_flux2_klein_9b_actionpatch_aug_full/2026-08-27_11-29-01"
export CKPT="${D}/checkpoints/weights/step_110000.pt"
STATS="${D}/dataset_stats.json"
export COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-9B/flux-2-klein-base-9b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-9b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-8B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=100 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"

TASKS="beat_block_hammer grab_roller move_stapler_pad open_microwave place_can_basket place_container_plate place_empty_cup place_fan place_mouse_pad press_stapler put_object_cabinet rotate_qrcode stack_blocks_two stack_bowls_three"
GLIST=""
for t in ${TASKS}; do GLIST="${GLIST} ${t}:demo_clean ${t}:demo_randomized"; done
export GLIST

echo "[launcher] aug9b110k seed=${SEED} groups=$(echo ${GLIST} | wc -w) nodes=${SLURM_JOB_NODELIST}"
srun --ntasks=2 --ntasks-per-node=1 --gres=gpu:8 --export=ALL bash exploration/action_img_patch/eval_100ep_worker.sh
echo "[launcher] aug9b110k seed=${SEED} COMPLETE"
