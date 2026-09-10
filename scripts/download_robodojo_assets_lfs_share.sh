#!/usr/bin/env bash
set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
SHARE_ROOT=/storage/yukaichengLab/share/datasets
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
ASSET_REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.modelscope_robodojo_repo
TARGET_LINK=/storage/yukaichengLab/share/datasets/RoboDojo_Assets
PARTIAL_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_asset_partials
SOURCE_URL=https://www.modelscope.cn/datasets/RoboDojo-Benchmark/RoboDojo.git
EXPECTED_FILES=15367
# Robots/Object/Material/Eval_Layout are the obvious ones; the eval client also
# needs Background (dome-light HDR), Sensor (camera USDs), Room (every scene
# layout references Simple_Room_nolight etc.) and Traj.
INCLUDE_PATHS='Assets/Robots/**,Assets/Object/**,Assets/Material/**,Assets/Eval_Layout/**,Assets/Background/**,Assets/Sensor/**,Assets/Room/**,Assets/Traj/**'

mkdir -p "${SHARE_ROOT}" "${STATE_DIR}" "${PARTIAL_ROOT}"

exec 9>"${STATE_DIR}/asset_lfs_download.lock"
if ! flock -n 9; then
  echo "Another RoboDojo Git LFS asset download is already running" >&2
  exit 1
fi

if [[ -e "${ASSET_REPO}" && ! -d "${ASSET_REPO}/.git" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "${ASSET_REPO}" "${PARTIAL_ROOT}/asset_repo.${stamp}"
fi

if [[ ! -d "${ASSET_REPO}/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone \
    --depth 1 \
    --sparse \
    "${SOURCE_URL}" \
    "${ASSET_REPO}"
fi

GIT_LFS_SKIP_SMUDGE=1 git -C "${ASSET_REPO}" sparse-checkout set Assets
git -C "${ASSET_REPO}" lfs install --local >/dev/null

download_ok=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "[asset-lfs] pull attempt ${attempt}"
  if git -C "${ASSET_REPO}" lfs pull \
    --include="${INCLUDE_PATHS}" \
    --exclude=""; then
    pointer_file=$(find \
      "${ASSET_REPO}/Assets/Robots" \
      "${ASSET_REPO}/Assets/Object" \
      "${ASSET_REPO}/Assets/Material" \
      "${ASSET_REPO}/Assets/Eval_Layout" \
      "${ASSET_REPO}/Assets/Background" \
      "${ASSET_REPO}/Assets/Sensor" \
      "${ASSET_REPO}/Assets/Room" \
      "${ASSET_REPO}/Assets/Traj" \
      -type f -size -300c \
      -exec grep -Il '^version https://git-lfs.github.com/spec/v1$' {} + \
      2>/dev/null | head -n 1 || true)
    file_count=$(find \
      "${ASSET_REPO}/Assets/Robots" \
      "${ASSET_REPO}/Assets/Object" \
      "${ASSET_REPO}/Assets/Material" \
      "${ASSET_REPO}/Assets/Eval_Layout" \
      "${ASSET_REPO}/Assets/Background" \
      "${ASSET_REPO}/Assets/Sensor" \
      "${ASSET_REPO}/Assets/Room" \
      "${ASSET_REPO}/Assets/Traj" \
      -type f | wc -l)
    if [[ -z "${pointer_file}" && "${file_count}" -eq "${EXPECTED_FILES}" ]]; then
      download_ok=1
      break
    fi
    echo "[asset-lfs] incomplete after attempt ${attempt}: files=${file_count}, pointer=${pointer_file:-none}" >&2
  fi
  sleep $((attempt * 5))
done

if [[ "${download_ok}" -ne 1 ]]; then
  echo "RoboDojo Git LFS asset download failed exact validation" >&2
  exit 1
fi

# Integrity check of exactly the four asset trees: every LFS-tracked file must
# hash to its pointer oid (sha256).  `git lfs fsck` is unsuitable here: its
# pointer pass flags upstream blobs committed as raw files, and its object pass
# walks data/ and ckpt/ objects that this sparse checkout never fetches.
verify_dir=$(mktemp -d)
git -C "${ASSET_REPO}" lfs ls-files -l -I "${INCLUDE_PATHS}" \
  | awk '{print $1 "  " $3}' >"${verify_dir}/expected.txt"
# Files we deliberately rewrote after download (online Omniverse MDL references ->
# local mirror, see share/datasets/ov_materials_mirror) keep their pristine copy as
# <file>.orig_online_mdl; verify that copy instead of the rewritten file.
python3 - "${verify_dir}/expected.txt" "${ASSET_REPO}" <<'PY'
import os, sys
path, repo = sys.argv[1], sys.argv[2]
lines = open(path).read().splitlines()
out = []
for line in lines:
    digest, rel = line.split("  ", 1)
    if os.path.exists(os.path.join(repo, rel + ".orig_online_mdl")):
        rel = rel + ".orig_online_mdl"
    out.append(f"{digest}  {rel}")
open(path, "w").write("\n".join(out) + "\n")
PY
expected_count=$(wc -l <"${verify_dir}/expected.txt")
[[ "${expected_count}" -gt 0 ]] || { echo "No LFS files matched ${INCLUDE_PATHS}" >&2; exit 1; }
split -n "l/8" "${verify_dir}/expected.txt" "${verify_dir}/chunk_"
if ! (cd "${ASSET_REPO}" && ls "${verify_dir}"/chunk_* | xargs -P 8 -I{} sha256sum -c --quiet {}); then
  echo "RoboDojo asset sha256 verification failed" >&2
  rm -rf "${verify_dir}"
  exit 1
fi
echo "[asset-lfs] sha256 verified ${expected_count} LFS files"
rm -rf "${verify_dir}"

if [[ -L "${TARGET_LINK}" ]]; then
  current_target=$(readlink -f "${TARGET_LINK}")
  if [[ "${current_target}" != "${ASSET_REPO}/Assets" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${TARGET_LINK}" "${PARTIAL_ROOT}/RoboDojo_Assets.link.${stamp}"
    ln -s "${ASSET_REPO}/Assets" "${TARGET_LINK}"
  fi
elif [[ -e "${TARGET_LINK}" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "${TARGET_LINK}" "${PARTIAL_ROOT}/RoboDojo_Assets.${stamp}"
  ln -s "${ASSET_REPO}/Assets" "${TARGET_LINK}"
else
  ln -s "${ASSET_REPO}/Assets" "${TARGET_LINK}"
fi

repo_assets="${ROBODOJO_ROOT}/Assets"
if [[ -L "${repo_assets}" ]]; then
  current_target=$(readlink -f "${repo_assets}")
  if [[ "${current_target}" != "${ASSET_REPO}/Assets" ]]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    mv "${repo_assets}" "${ROBODOJO_ROOT}/Assets.partial.${stamp}"
    ln -s "${TARGET_LINK}" "${repo_assets}"
  fi
elif [[ -e "${repo_assets}" ]]; then
  stamp=$(date +%Y%m%d_%H%M%S)
  mv "${repo_assets}" "${ROBODOJO_ROOT}/Assets.partial.${stamp}"
  ln -s "${TARGET_LINK}" "${repo_assets}"
else
  ln -s "${TARGET_LINK}" "${repo_assets}"
fi

printf 'source=ModelScopeGitLFS:RoboDojo-Benchmark/RoboDojo\ncommit=%s\nfiles=%s\n' \
  "$(git -C "${ASSET_REPO}" rev-parse HEAD)" \
  "${EXPECTED_FILES}" \
  >"${STATE_DIR}/asset_download.complete"
du -sh "${ASSET_REPO}/Assets" >"${STATE_DIR}/asset_size.txt"
find \
  "${ASSET_REPO}/Assets/Robots" \
  "${ASSET_REPO}/Assets/Object" \
  "${ASSET_REPO}/Assets/Material" \
  "${ASSET_REPO}/Assets/Eval_Layout" \
  "${ASSET_REPO}/Assets/Background" \
  "${ASSET_REPO}/Assets/Sensor" \
  "${ASSET_REPO}/Assets/Room" \
  "${ASSET_REPO}/Assets/Traj" \
  -type f -printf '%s\t%p\n' \
  >"${STATE_DIR}/asset_files.tsv"
