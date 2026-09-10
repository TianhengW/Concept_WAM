#!/usr/bin/env bash
set -euo pipefail

IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
DATA_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/data
STATE_DIR=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.robodojo_state
SOURCE_REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.modelscope_robodojo_repo
SOURCE_DATA=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.modelscope_robodojo_repo/data/RoboDojo
TARGET_DIR=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/RoboDojo
SHARD_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.robodojo_task_shards
PARTIAL_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.robodojo_cli_partials
TASK_STATE=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.robodojo_state/tasks
MODELSCOPE_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/modelscope

mkdir -p \
  "${DATA_ROOT}" \
  "${STATE_DIR}" \
  "${TARGET_DIR}" \
  "${SHARD_ROOT}" \
  "${PARTIAL_ROOT}" \
  "${TASK_STATE}"

mapfile -t tasks < <(
  find "${SOURCE_DATA}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)

if [[ "${#tasks[@]}" -ne 35 ]]; then
  echo "Unexpected task inventory: ${#tasks[@]}, expected 35" >&2
  exit 1
fi

download_one_task() {
  local task_name="$1"
  local shard_dir="${SHARD_ROOT}/${task_name}"
  local shard_task="${shard_dir}/data/RoboDojo/${task_name}"
  local target_task="${TARGET_DIR}/${task_name}"
  local task_marker="${TASK_STATE}/${task_name}.complete"
  local hdf5_count
  local mp4_count
  local stamp

  if [[ -f "${task_marker}" && -d "${target_task}" ]]; then
    hdf5_count=$(find "${target_task}" -type f -name '*.hdf5' | wc -l)
    mp4_count=$(find "${target_task}" -type f -name '*.mp4' | wc -l)
    if [[ "${hdf5_count}" -eq 100 && "${mp4_count}" -eq 300 ]]; then
      echo "[task:${task_name}] already complete"
      return 0
    fi
  fi

  mkdir -p "${shard_dir}"

  "${MODELSCOPE_BIN}" download \
    --dataset RoboDojo-Benchmark/RoboDojo \
    --include "data/RoboDojo/${task_name}/**" \
    --local_dir "${shard_dir}" \
    --max-workers 8

  hdf5_count=$(find "${shard_task}" -type f -name '*.hdf5' | wc -l)
  mp4_count=$(find "${shard_task}" -type f -name '*.mp4' | wc -l)

  if [[ "${hdf5_count}" -ne 100 || "${mp4_count}" -ne 300 ]]; then
    echo "[task:${task_name}] incomplete: hdf5=${hdf5_count}, mp4=${mp4_count}" >&2
    return 1
  fi

  if find "${shard_task}" -type f -size -1024c -print -quit | grep -q .; then
    echo "[task:${task_name}] small file remains" >&2
    return 1
  fi

  if [[ -e "${target_task}" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${target_task}" "${PARTIAL_ROOT}/${task_name}.${stamp}"
  fi

  mv "${shard_task}" "${target_task}"
  printf 'complete\n' > "${task_marker}"
  echo "[task:${task_name}] complete"
}

export \
  DATA_ROOT \
  IMAGEWAM_ROOT \
  MODELSCOPE_BIN \
  PARTIAL_ROOT \
  SHARD_ROOT \
  STATE_DIR \
  TARGET_DIR \
  TASK_STATE
export -f download_one_task

printf '%s\0' "${tasks[@]}" | \
  xargs -0 -n 1 -P 8 bash -c 'download_one_task "$1"' _

hdf5_count=$(find "${TARGET_DIR}" -type f -name '*.hdf5' | wc -l)
mp4_count=$(find "${TARGET_DIR}" -type f -name '*.mp4' | wc -l)
task_count=$(find "${TASK_STATE}" -type f -name '*.complete' | wc -l)

if [[ "${task_count}" -ne 35 || "${hdf5_count}" -ne 3500 || "${mp4_count}" -ne 10500 ]]; then
  echo "Incomplete training set: tasks=${task_count}, hdf5=${hdf5_count}, mp4=${mp4_count}" >&2
  exit 1
fi

if find "${TARGET_DIR}" -type f -size -1024c -print -quit | grep -q .; then
  echo "Small files remain; possible unresolved LFS pointers" >&2
  exit 1
fi

if find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 \
  \( -name 'RoboDojo_depth' -o -name 'RoboDojo_real' -o -name 'RoboDojo_lerobot_*' \) \
  -print -quit | grep -q .; then
  echo "Unexpected excluded dataset directory found" >&2
  exit 1
fi

printf 'source=ModelScope:RoboDojo-Benchmark/RoboDojo\ncommit=%s\n' \
  "$(git -C "${SOURCE_REPO}" rev-parse HEAD)" \
  > "${TARGET_DIR}/.download_complete"
printf 'tasks=%s\nhdf5=%s\nmp4=%s\n' \
  "${task_count}" "${hdf5_count}" "${mp4_count}" \
  > "${STATE_DIR}/training_counts.txt"
find "${TARGET_DIR}" -type f -printf '%s\t%p\n' \
  > "${STATE_DIR}/training_files.tsv"
du -sh "${TARGET_DIR}" > "${STATE_DIR}/training_size.txt"
printf 'complete\n' > "${STATE_DIR}/training_download.complete"
