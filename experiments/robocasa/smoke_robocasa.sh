#!/usr/bin/env bash
# RoboCasa365 pretrain smoke test on a single H200.
# Verifies v2.1 loading / 3-cam compact layout / loss, and measures peak GPU
# memory to tune batch_size toward >=80% of 140G (>=115000 MiB).
# Usage: SMOKE_BATCH=12 SMOKE_GPU=4 bash smoke_robocasa.sh
set -euo pipefail

REPO=/mnt/auwomo-data/wangtianheng/codes/action-as-patch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_robocasa.sh"

VENV="${SMOKE_VENV:-/root/.venv_c2r_smoke}"
source "${VENV}/bin/activate"

cd "${REPO}"
GPU="${SMOKE_GPU:-4}"
BATCH="${SMOKE_BATCH:-12}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TASK=robocasa_flux2_klein_4b_actionpatch_pretrain

LOG="${SCRIPT_DIR}/smoke_robocasa_b${BATCH}.log"
MEMLOG="${SCRIPT_DIR}/smoke_robocasa_b${BATCH}.mem"

( while true; do nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done > "${MEMLOG}" ) &
MEMPID=$!
trap 'kill ${MEMPID} 2>/dev/null || true' EXIT

echo "[smoke] batch=${BATCH} gpu=${GPU} logging to ${LOG}"
bash scripts/flux2/train_flux2_klein_imagewam.sh 1 \
  batch_size="${BATCH}" \
  max_steps=10 log_every=1 \
  num_workers=8 \
  save_every=1000000 \
  wandb.enabled=false \
  2>&1 | tee "${LOG}"

kill ${MEMPID} 2>/dev/null || true
PEAK=$(sort -n "${MEMLOG}" | tail -1)
echo "[smoke] PEAK GPU MEM = ${PEAK} MiB / 143771 MiB ($(python3 -c "print(f'{${PEAK}/143771*100:.1f}')")%)"
echo "[smoke] target >=80% (>=115000 MiB). Adjust SMOKE_BATCH and re-run if needed."
