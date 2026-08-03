#!/usr/bin/env bash
# Resubmit v3_4node, resuming EXACTLY from the latest saved training state
# (model + optimizer + scheduler + RNG + step). Auto-detects the newest
# checkpoints/state/step_XXXXXX under the existing run dir and continues writing
# into that SAME run dir (so checkpoints stay in one place).
#
# Usage:  bash v3/scripts/resubmit_v3_resume.sh
set -euo pipefail
source /etc/profile.d/slurm.sh 2>/dev/null || true

REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM
RUN_DIR="$REPO/v3/runs/robotwin_v3/2026-07-31_13-35-24"
STATE_ROOT="$RUN_DIR/checkpoints/state"

# newest step_* state directory (accelerate save_state = full resumable state)
LATEST_STATE=$(ls -d "$STATE_ROOT"/step_* 2>/dev/null | sort -V | tail -n1)
if [[ -z "${LATEST_STATE:-}" || ! -f "$LATEST_STATE/latest" ]]; then
  echo "ERROR: no complete state checkpoint found under $STATE_ROOT" >&2
  exit 1
fi

export RESUME="$LATEST_STATE"
export RUN_ID="2026-07-31_13-35-24"   # continue in the same run dir

echo "[resubmit] resuming from: $RESUME"
echo "[resubmit] run dir:       $RUN_DIR"
sbatch "$REPO/v3/scripts/sbatch_v3_4node.sh"
