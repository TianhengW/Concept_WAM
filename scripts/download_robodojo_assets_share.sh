#!/usr/bin/env bash
set -euo pipefail

IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
SHARE_ROOT=/storage/yukaichengLab/share/datasets
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
SOURCE_REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.modelscope_robodojo_repo
TARGET_ROOT=/storage/yukaichengLab/share/datasets/RoboDojo_Assets
SHARD_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_asset_shards
PARTIAL_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_asset_partials
COMPONENT_STATE=/storage/yukaichengLab/share/datasets/.robodojo_state/asset_components
COMPONENT_LOG=/storage/yukaichengLab/share/datasets/.robodojo_state/asset_logs
MODELSCOPE_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/modelscope

COMPONENTS=(Robots Object Material Eval_Layout)

mkdir -p \
  "${SHARE_ROOT}" \
  "${STATE_DIR}" \
  "${TARGET_ROOT}" \
  "${SHARD_ROOT}" \
  "${PARTIAL_ROOT}" \
  "${COMPONENT_STATE}" \
  "${COMPONENT_LOG}"

exec 9>"${STATE_DIR}/asset_download.lock"
if ! flock -n 9; then
  echo "Another RoboDojo asset download is already running" >&2
  exit 1
fi

for component in "${COMPONENTS[@]}"; do
  if [[ ! -d "${SOURCE_REPO}/Assets/${component}" ]]; then
    echo "Missing asset metadata: ${SOURCE_REPO}/Assets/${component}" >&2
    exit 1
  fi
done

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

download_one_component() {
  local component="$1"
  local expected_tree="${SOURCE_REPO}/Assets/${component}"
  local target_tree="${TARGET_ROOT}/${component}"
  local shard_dir="${SHARD_ROOT}/${component}"
  local shard_tree="${SHARD_ROOT}/${component}/Assets/${component}"
  local marker="${COMPONENT_STATE}/${component}.complete"
  local component_log="${COMPONENT_LOG}/${component}.log"
  local attempt
  local download_ok=0
  local stamp

  if validate_tree "${expected_tree}" "${target_tree}"; then
    printf 'complete\n' >"${marker}"
    echo "[asset:${component}] already complete"
    return 0
  fi

  mkdir -p "${shard_dir}/.tmp"

  for attempt in 1 2 3 4 5; do
    echo "[asset:${component}] download attempt ${attempt}"
    if (
      cd "${shard_dir}"
      MODELSCOPE_CACHE="${shard_dir}/.cache" \
      TMPDIR="${shard_dir}/.tmp" \
      "${MODELSCOPE_BIN}" download \
        --dataset RoboDojo-Benchmark/RoboDojo \
        --include "Assets/${component}/**" \
        --local_dir "${shard_dir}" \
        --max-workers 8
    ) >"${component_log}" 2>&1; then
      if validate_tree "${expected_tree}" "${shard_tree}"; then
        download_ok=1
        break
      fi
      echo "[asset:${component}] attempt ${attempt} returned an incomplete tree" >&2
    fi
    tail -n 20 "${component_log}" >&2 || true
    sleep $((attempt * 10))
  done

  if [[ "${download_ok}" -ne 1 ]]; then
    echo "[asset:${component}] failed after 5 attempts; see ${component_log}" >&2
    return 1
  fi

  validate_tree "${expected_tree}" "${shard_tree}"

  if [[ -e "${target_tree}" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${target_tree}" "${PARTIAL_ROOT}/${component}.${stamp}"
  fi

  mv "${shard_tree}" "${target_tree}"
  printf 'complete\n' >"${marker}"
  echo "[asset:${component}] complete"
}

export \
  COMPONENT_LOG \
  COMPONENT_STATE \
  MODELSCOPE_BIN \
  PARTIAL_ROOT \
  SHARD_ROOT \
  SOURCE_REPO \
  TARGET_ROOT
export -f download_one_component validate_tree

printf '%s\0' "${COMPONENTS[@]}" | \
  xargs -0 -n 1 -P 4 bash -c 'download_one_component "$1"' _

for component in "${COMPONENTS[@]}"; do
  validate_tree \
    "${SOURCE_REPO}/Assets/${component}" \
    "${TARGET_ROOT}/${component}"
done

repo_assets="${ROBODOJO_ROOT}/Assets"
if [[ -L "${repo_assets}" ]]; then
  current_target=$(readlink -f "${repo_assets}")
  if [[ "${current_target}" != "${TARGET_ROOT}" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${repo_assets}" "${ROBODOJO_ROOT}/Assets.partial.${stamp}"
    ln -s "${TARGET_ROOT}" "${repo_assets}"
  fi
elif [[ -e "${repo_assets}" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "${repo_assets}" "${ROBODOJO_ROOT}/Assets.partial.${stamp}"
  ln -s "${TARGET_ROOT}" "${repo_assets}"
else
  ln -s "${TARGET_ROOT}" "${repo_assets}"
fi

find "${TARGET_ROOT}" -type f -printf '%s\t%p\n' \
  >"${STATE_DIR}/asset_files.tsv"
du -sh "${TARGET_ROOT}" >"${STATE_DIR}/asset_size.txt"
printf 'complete\n' >"${STATE_DIR}/asset_download.complete"
