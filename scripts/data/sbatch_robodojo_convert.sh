#!/usr/bin/env bash
# CPU-only: convert RoboDojo raw hdf5 -> fastwam-style LeRobot v2.1 (batch, resumable).
# Usage: sbatch scripts/data/sbatch_robodojo_convert.sh --out-root <dir> [--tasks complete|all|a,b] [--every N] [--limit K]
# Must run on a compute node: the login node's cgroup OOM-kills parallel workers.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH -x nv-h100-029,nv-h100-007
#SBATCH -J robodojo_convert
#SBATCH -o slurm_logs/%x_%j.out
set -euo pipefail
cd /storage/yukaichengLab/mazijian/wth/ImageWAM
mkdir -p slurm_logs
exec .venv/bin/python scripts/data/robodojo_raw_to_lerobot.py \
  --batch --workers "${CONVERT_WORKERS:-24}" \
  "$@"
