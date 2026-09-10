#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -x gnho006,gnho008,gnho009,gnho011,gnho016,gnho017,gnho018,gnho020,gnho031,gnho032,gnho033,gnho034,gnho036,gnho038,gnho041
#SBATCH -J vmse_batch
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 1:40:00
set -e
REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:$REPO/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
python exploration/action_img_patch/compare_video_mse_batch.py
