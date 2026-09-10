#!/bin/bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -x gnho006,gnho008,gnho009,gnho011,gnho016,gnho017,gnho018,gnho020,gnho031,gnho032,gnho033,gnho034,gnho035,gnho036,gnho038,gnho041
#SBATCH -J smoke_vl
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -t 0:40:00
set -e
REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM
cd "$REPO"
source .venv_v3/bin/activate
export PYTHONPATH="$REPO:$REPO/src:/storage/yukaichengLab/mazijian/wth/flux2/src:/storage/yukaichengLab/mazijian/wth/flux2:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python exploration/action_img_patch/smoke_vl.py
