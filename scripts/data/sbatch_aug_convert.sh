#!/usr/bin/env bash
# CPU-only: convert Robotwin_aug raw hdf5 -> fastwam LeRobot v2.1 (batch, resumable)
# Extra args (e.g. --refresh-parquet) pass through to the converter.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041
#SBATCH -J aug_convert
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
cd /storage/yukaichengLab/mazijian/wth/ImageWAM
exec .venv/bin/python scripts/data/robotwin_raw_to_lerobot.py \
  --batch --workers 24 \
  --out-root /storage/yukaichengLab/mazijian/wth/Robotwin_aug_lerobot \
  "$@"
