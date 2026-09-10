#!/usr/bin/env bash
set -euo pipefail

SHARE_ROOT=/storage/yukaichengLab/share/datasets
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
SOURCE_REPO=/storage/yukaichengLab/share/datasets/.robodojo_meta
SOURCE_DATA=/storage/yukaichengLab/share/datasets/.robodojo_meta/data/RoboDojo
TARGET_DIR=/storage/yukaichengLab/share/datasets/RoboDojo
SHARD_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_task_shards
PARTIAL_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_cli_partials
TASK_STATE=/storage/yukaichengLab/share/datasets/.robodojo_state/tasks
TASK_LOG=/storage/yukaichengLab/share/datasets/.robodojo_state/task_logs
MODELSCOPE_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/modelscope
SOURCE_URL=https://www.modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo.git

mkdir -p \
  "${SHARE_ROOT}" \
  "${STATE_DIR}" \
  "${TARGET_DIR}" \
  "${SHARD_ROOT}" \
  "${PARTIAL_ROOT}" \
  "${TASK_STATE}" \
  "${TASK_LOG}"

exec 9>"${STATE_DIR}/training_download.lock"
if ! flock -n 9; then
  echo "Another RoboDojo training download is already running" >&2
  exit 1
fi

if [[ ! -d "${SOURCE_REPO}/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    --depth 1 \
    --filter=blob:none \
    --sparse \
    "${SOURCE_URL}" \
    "${SOURCE_REPO}"
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${SOURCE_REPO}" sparse-checkout set data/RoboDojo

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
  local attempt
  local download_ok=0
  local task_log="${TASK_LOG}/${task_name}.log"

  if [[ -f "${task_marker}" && -d "${target_task}" ]]; then
    hdf5_count=$(find "${target_task}" -type f -name '*.hdf5' | wc -l)
    mp4_count=$(find "${target_task}" -type f -name '*.mp4' | wc -l)
    if [[ "${hdf5_count}" -eq 100 && "${mp4_count}" -eq 300 ]]; then
      echo "[task:${task_name}] already complete"
      return 0
    fi
  fi

  mkdir -p "${shard_dir}/.tmp"

  for attempt in 1 2 3 4 5; do
    echo "[task:${task_name}] download attempt ${attempt}"
    if (
      cd "${shard_dir}"
      MODELSCOPE_CACHE="${shard_dir}/.cache" \
      TMPDIR="${shard_dir}/.tmp" \
      "${MODELSCOPE_BIN}" download \
        --dataset RoboDojo-Benchmark/RoboDojo \
        --include "data/RoboDojo/${task_name}/**" \
        --local_dir "${shard_dir}" \
        --max-workers 8
    ) >"${task_log}" 2>&1; then
      if [[ -d "${shard_task}" ]]; then
        hdf5_count=$(find "${shard_task}" -type f -name '*.hdf5' | wc -l)
        mp4_count=$(find "${shard_task}" -type f -name '*.mp4' | wc -l)
      else
        hdf5_count=0
        mp4_count=0
      fi
      if [[ "${hdf5_count}" -eq 100 && "${mp4_count}" -eq 300 ]] && \
        ! find "${shard_task}" -type f -size -1024c -print -quit | grep -q .; then
        download_ok=1
        break
      fi
      echo "[task:${task_name}] attempt ${attempt} returned incomplete: hdf5=${hdf5_count}, mp4=${mp4_count}" >&2
    fi
    tail -n 20 "${task_log}" >&2 || true
    sleep $((attempt * 10))
  done

  if [[ "${download_ok}" -ne 1 ]]; then
    echo "[task:${task_name}] download failed after 5 attempts; see ${task_log}" >&2
    return 1
  fi

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
  MODELSCOPE_BIN \
  PARTIAL_ROOT \
  SHARE_ROOT \
  SHARD_ROOT \
  STATE_DIR \
  TARGET_DIR \
  TASK_LOG \
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

if find "${SHARE_ROOT}" -mindepth 1 -maxdepth 1 \
  \( -name 'RoboDojo_depth' -o -name 'RoboDojo_real' \) \
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
