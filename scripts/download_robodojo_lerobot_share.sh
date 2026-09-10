#!/usr/bin/env bash
set -euo pipefail

SHARE_ROOT=/storage/yukaichengLab/share/datasets
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
SOURCE_REPO=/storage/yukaichengLab/share/datasets/.robodojo_meta
SHARD_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_lerobot_shards
PARTIAL_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_lerobot_partials
GROUP_STATE=/storage/yukaichengLab/share/datasets/.robodojo_state/lerobot_groups
GROUP_LOG=/storage/yukaichengLab/share/datasets/.robodojo_state/lerobot_logs
MODELSCOPE_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/modelscope
SOURCE_URL=https://www.modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo.git

DATASETS=(
  RoboDojo_ee_lerobot_v30_video
  RoboDojo_lerobot_v21_video
  RoboDojo_lerobot_v30_video
)

mkdir -p \
  "${SHARE_ROOT}" \
  "${STATE_DIR}" \
  "${SHARD_ROOT}" \
  "${PARTIAL_ROOT}" \
  "${GROUP_STATE}" \
  "${GROUP_LOG}"

exec 9>"${STATE_DIR}/lerobot_download.lock"
if ! flock -n 9; then
  echo "Another RoboDojo LeRobot download is already running" >&2
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

GIT_LFS_SKIP_SMUDGE=1 git -C "${SOURCE_REPO}" sparse-checkout set \
  data/RoboDojo \
  data/RoboDojo_ee_lerobot_v30_video \
  data/RoboDojo_lerobot_v21_video \
  data/RoboDojo_lerobot_v30_video

validate_tree() {
  local expected_root="$1"
  local actual_root="$2"
  local expected_count
  local actual_count
  local expected_file
  local relative_path
  local actual_file
  local expected_size
  local actual_size

  [[ -d "${expected_root}" && -d "${actual_root}" ]] || return 1

  expected_count=$(find "${expected_root}" -type f | wc -l)
  actual_count=$(find "${actual_root}" -type f | wc -l)
  [[ "${expected_count}" -eq "${actual_count}" ]] || return 1

  while IFS= read -r -d '' expected_file; do
    relative_path=${expected_file#"${expected_root}"/}
    actual_file="${actual_root}/${relative_path}"
    [[ -f "${actual_file}" ]] || return 1

    expected_size=$(awk '$1 == "size" {print $2; exit}' "${expected_file}")
    if [[ -z "${expected_size}" ]]; then
      expected_size=$(stat -c '%s' "${expected_file}")
    fi
    actual_size=$(stat -c '%s' "${actual_file}")
    [[ "${expected_size}" -eq "${actual_size}" ]] || return 1
  done < <(find "${expected_root}" -type f -print0)
}

download_one_group() {
  local spec="$1"
  local dataset=${spec%%|*}
  local relative_path=${spec#*|}
  local group_id
  local expected_tree="${SOURCE_REPO}/data/${dataset}/${relative_path}"
  local target_tree="${SHARE_ROOT}/${dataset}/${relative_path}"
  local shard_dir
  local shard_tree
  local marker
  local group_log
  local attempt
  local download_ok=0
  local stamp

  group_id=$(printf '%s' "${relative_path}" | tr '/.' '__')
  shard_dir="${SHARD_ROOT}/${dataset}/${group_id}"
  shard_tree="${shard_dir}/data/${dataset}/${relative_path}"
  marker="${GROUP_STATE}/${dataset}/${group_id}.complete"
  group_log="${GROUP_LOG}/${dataset}__${group_id}.log"

  if validate_tree "${expected_tree}" "${target_tree}"; then
    mkdir -p "$(dirname "${marker}")"
    printf 'complete\n' >"${marker}"
    echo "[lerobot:${dataset}:${relative_path}] already complete"
    return 0
  fi

  mkdir -p "${shard_dir}/.tmp" "$(dirname "${marker}")"

  for attempt in 1 2 3 4 5; do
    echo "[lerobot:${dataset}:${relative_path}] download attempt ${attempt}"
    if (
      cd "${shard_dir}"
      MODELSCOPE_CACHE="${shard_dir}/.cache" \
      TMPDIR="${shard_dir}/.tmp" \
      "${MODELSCOPE_BIN}" download \
        --dataset RoboDojo-Benchmark/RoboDojo \
        --include "data/${dataset}/${relative_path}/**" \
        --local_dir "${shard_dir}" \
        --max-workers 6
    ) >"${group_log}" 2>&1; then
      if validate_tree "${expected_tree}" "${shard_tree}"; then
        download_ok=1
        break
      fi
      echo "[lerobot:${dataset}:${relative_path}] attempt ${attempt} returned an incomplete tree" >&2
    fi
    tail -n 20 "${group_log}" >&2 || true
    sleep $((attempt * 10))
  done

  if [[ "${download_ok}" -ne 1 ]]; then
    echo "[lerobot:${dataset}:${relative_path}] failed after 5 attempts; see ${group_log}" >&2
    return 1
  fi

  validate_tree "${expected_tree}" "${shard_tree}"

  if [[ -e "${target_tree}" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p "${PARTIAL_ROOT}/${dataset}"
    mv "${target_tree}" "${PARTIAL_ROOT}/${dataset}/${group_id}.${stamp}"
  fi

  mkdir -p "$(dirname "${target_tree}")"
  mv "${shard_tree}" "${target_tree}"
  printf 'complete\n' >"${marker}"
  echo "[lerobot:${dataset}:${relative_path}] complete"
}

group_specs=()

for dataset in RoboDojo_ee_lerobot_v30_video RoboDojo_lerobot_v30_video; do
  group_specs+=("${dataset}|data")
  group_specs+=("${dataset}|meta")
  while IFS= read -r camera; do
    group_specs+=("${dataset}|videos/${camera}")
  done < <(
    find "${SOURCE_REPO}/data/${dataset}/videos" \
      -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  )
done

dataset=RoboDojo_lerobot_v21_video
while IFS= read -r chunk; do
  group_specs+=("${dataset}|data/${chunk}")
done < <(
  find "${SOURCE_REPO}/data/${dataset}/data" \
    -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
)
group_specs+=("${dataset}|meta")
while IFS= read -r camera_dir; do
  camera_dir=${camera_dir#"${SOURCE_REPO}/data/${dataset}/"}
  group_specs+=("${dataset}|${camera_dir}")
done < <(
  find "${SOURCE_REPO}/data/${dataset}/videos" \
    -mindepth 2 -maxdepth 2 -type d | sort
)

if [[ "${#group_specs[@]}" -ne 31 ]]; then
  echo "Unexpected LeRobot group inventory: ${#group_specs[@]}, expected 31" >&2
  exit 1
fi

export \
  GROUP_LOG \
  GROUP_STATE \
  MODELSCOPE_BIN \
  PARTIAL_ROOT \
  SHARE_ROOT \
  SHARD_ROOT \
  SOURCE_REPO
export -f download_one_group validate_tree

printf '%s\0' "${group_specs[@]}" | \
  xargs -0 -n 1 -P 6 bash -c 'download_one_group "$1"' _

summary_file="${STATE_DIR}/lerobot_counts.txt"
: >"${summary_file}"

for dataset in "${DATASETS[@]}"; do
  expected_root="${SOURCE_REPO}/data/${dataset}"
  target_root="${SHARE_ROOT}/${dataset}"

  if ! validate_tree "${expected_root}" "${target_root}"; then
    echo "Final exact validation failed for ${dataset}" >&2
    exit 1
  fi

  file_count=$(find "${target_root}" -type f | wc -l)
  disk_bytes=$(du -sb "${target_root}" | awk '{print $1}')
  printf '%s files=%s bytes=%s\n' "${dataset}" "${file_count}" "${disk_bytes}" \
    >>"${summary_file}"
done

find \
  "${SHARE_ROOT}/RoboDojo_ee_lerobot_v30_video" \
  "${SHARE_ROOT}/RoboDojo_lerobot_v21_video" \
  "${SHARE_ROOT}/RoboDojo_lerobot_v30_video" \
  -type f -printf '%s\t%p\n' \
  >"${STATE_DIR}/lerobot_files.tsv"

printf 'source=ModelScope:RoboDojo-Benchmark/RoboDojo\ncommit=%s\n' \
  "$(git -C "${SOURCE_REPO}" rev-parse HEAD)" \
  >"${STATE_DIR}/lerobot_download.complete"
