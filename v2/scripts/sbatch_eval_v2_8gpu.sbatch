#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v2_eval_step5000
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs/%x_%j.out
set -euo pipefail
mkdir -p /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs
# Full eval: all ~53 tasks x 10 episodes x [clean,random] on 8 GPUs.
NUM_GPUS=8 MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}" EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}" \
  PHASES="${PHASES:-[clean,random]}" TASK_NAME="${TASK_NAME:-null}" \
  bash /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/scripts/eval_v2_robotwin.sh "$@"
