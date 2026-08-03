#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 90
#SBATCH -x gnho032,nv-h100-029,nv-h100-007
#SBATCH -J v3eval
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v3/eval_logs/v3eval_%j.out
set -euo pipefail

cd /storage/yukaichengLab/mazijian/wth/ImageWAM

export EXP_PATH=/storage/yukaichengLab/mazijian/wth/ImageWAM/v3/runs/robotwin_v3/2026-07-31_13-35-24
export EVAL_TRAIN_STEP=005000
export NUM_GPUS=1
export MAX_TASKS_PER_GPU=1
export EVAL_NUM_EPISODES=1
export PHASES="[clean]"
export TASK_NAME=stack_blocks_two

bash v3/scripts/eval_v3_robotwin.sh
