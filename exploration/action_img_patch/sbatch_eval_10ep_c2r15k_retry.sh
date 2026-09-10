#!/bin/bash
# 100ep dual-phase eval of aug-mixed run ckpt step_015000 (50 tasks x clean+random x 100ep).
# Cloned from sbatch_eval_100ep_2node.sh; only CKPT/STATS/dirs changed. seed=43 same as baseline.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH -J eval_c2r15k_retry
#SBATCH --exclude=gnho008,gnho016,gnho036,gnho038
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 8:00:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"

export SEED="${SEED:-43}"
export REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
export EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"
export LOGDIR="slurm_logs/eval_10ep_c2r015000"
export RESDIR="${REPO_ROOT}/evaluate_results/robotwin/eval_10ep_c2r015000"
mkdir -p "${LOGDIR}" "${RESDIR}"

D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_c2r/2026-08-27_04-12-13"
export CKPT="${D}/checkpoints/weights/step_015000.pt"
STATS="${D}/dataset_stats.json"
export COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"

TASKS="move_stapler_pad place_mouse_pad adjust_bottle place_burger_fries press_stapler"
GLIST=""
for t in ${TASKS}; do GLIST="${GLIST} ${t}:demo_clean ${t}:demo_randomized"; done
export GLIST

echo "[launcher] c2r30k seed=${SEED} groups=$(echo ${GLIST} | wc -w) nodes=${SLURM_JOB_NODELIST}"
bash exploration/action_img_patch/eval_2gpu_worker.sh
echo "[launcher] c2r30k seed=${SEED} COMPLETE"
