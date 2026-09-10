#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J smoke_mix
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
cd /storage/yukaichengLab/mazijian/wth/ImageWAM
export PYTHONPATH="/storage/yukaichengLab/mazijian/wth/ImageWAM/src:/storage/yukaichengLab/mazijian/wth/ImageWAM"
exec .venv/bin/python scripts/data/smoke_mix_dataset.py
