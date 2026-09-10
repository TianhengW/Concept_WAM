#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/storage/yukaichengLab/mazijian/wth/ImageWAM/scripts
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
BASE_SCRIPT=${SCRIPT_DIR}/download_robodojo_base_share.sh
LEROBOT_SCRIPT=${SCRIPT_DIR}/download_robodojo_lerobot_share.sh

mkdir -p "${STATE_DIR}"

exec 9>"${STATE_DIR}/all_training_download.lock"
if ! flock -n 9; then
  echo "Another complete RoboDojo training download is already running" >&2
  exit 1
fi

base_rc=0
lerobot_rc=0

"${BASE_SCRIPT}" || base_rc=$?
"${LEROBOT_SCRIPT}" || lerobot_rc=$?

printf 'base_rc=%s\nlerobot_rc=%s\n' "${base_rc}" "${lerobot_rc}" \
  >"${STATE_DIR}/all_training_download.status"

if [[ "${base_rc}" -ne 0 || "${lerobot_rc}" -ne 0 ]]; then
  echo "RoboDojo training download incomplete: base_rc=${base_rc}, lerobot_rc=${lerobot_rc}" >&2
  exit 1
fi

printf 'complete\n' >"${STATE_DIR}/all_training_download.complete"
