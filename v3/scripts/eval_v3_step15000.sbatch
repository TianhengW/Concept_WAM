#!/usr/bin/env bash
# Full RoboTwin evaluation of the v3 (VL LatentReasoner) step_015000 checkpoint.
# 8 GPUs, all 50 tasks, 50 episodes each, clean+random phases.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=0
#SBATCH -t 2-00:00
#SBATCH -x gnho032,nv-h100-029,nv-h100-007
#SBATCH -J v3eval_15k
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v3/eval_logs/v3eval_15k_%j.out
set -euo pipefail

cd /storage/yukaichengLab/mazijian/wth/ImageWAM

export EXP_PATH=/storage/yukaichengLab/mazijian/wth/ImageWAM/v3/runs/robotwin_v3/2026-07-31_13-35-24
export EVAL_TRAIN_STEP=015000
export NUM_GPUS=8
export MAX_TASKS_PER_GPU=1
export EVAL_NUM_EPISODES=50
export PHASES="[clean,random]"
export TASK_NAME=null
# watchdog: kill a worker after 30 min with no stdout progress (records failed, run continues)
export ROBOTWIN_TASK_STALL_TIMEOUT_S=1800

bash v3/scripts/eval_v3_robotwin.sh
