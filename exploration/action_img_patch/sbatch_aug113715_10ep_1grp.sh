#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 03:00:00
set -uo pipefail
export PATH="/soft/slurm/bin:${PATH}"
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
EVAL_ENV_ROOT="/storage/yukaichengLab/mazijian/wth/Automodel/third_party/ImageWAM"
cd "${REPO_ROOT}"
source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:/soft/slurm/bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
D="${REPO_ROOT}/runs/robotwin_flux2_klein_4b_actionpatch_aug_full/2026-08-26_08-15-54"
CKPT="${D}/checkpoints/weights/step_113715.pt"; STATS="${D}/dataset_stats.json"
LOGDIR="slurm_logs/eval_10ep_aug113715"; RESDIR="${REPO_ROOT}/evaluate_results/robotwin/eval_10ep_aug113715"
COMMON="task=robotwin_flux2_klein_4b_actionpatch_sub20 EVALUATION.dataset_stats_path=${STATS} model.flux2_src_path=/storage/yukaichengLab/mazijian/wth/flux2 model.flux2_model_path=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors model.ae_model_path=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors model.variant=klein-base-4b model.qwen3_model_spec=/storage/yukaichengLab/share/model/Qwen3-4B model.load_text_encoder=true model.pack_proprio_after_text=true EVALUATION.action_horizon=16 EVALUATION.replan_steps=16 model.proprio_dim=14 EVALUATION.eval_num_episodes=10 EVALUATION.skip_get_obs_within_replan=true EVALUATION.robotwin_camera_layout=compact_288x256"
task=${GRP%%:*}; cfg=${GRP##*:}
timeout -s KILL 9000 python experiments/robotwin/eval_robotwin_single.py ckpt="${CKPT}" gpu_id=0 seed=43 EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} EVALUATION.output_dir="${RESDIR}/${task}_${cfg}" ${COMMON} > "${LOGDIR}/${task}_${cfg}.log" 2>&1 && echo "OK $task $cfg" || echo "FAIL $task $cfg"
