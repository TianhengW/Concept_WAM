#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v2_eval_smoke
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs/%x_%j.out
set -euo pipefail
mkdir -p /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs
# Minimal wiring test: 1 GPU, 1 task, 1 episode, clean phase only.
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 EVAL_NUM_EPISODES=1 PHASES="[clean]" TASK_NAME=click_bell \
  bash /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/scripts/eval_v2_robotwin.sh "$@"
