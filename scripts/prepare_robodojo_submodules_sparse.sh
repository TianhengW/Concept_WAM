#!/usr/bin/env bash
set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
ISAACLAB_DIR=${ROBODOJO_ROOT}/third_party/IsaacLab
CUROBO_DIR=${ROBODOJO_ROOT}/third_party/curobo
ISAACLAB_COMMIT=afca7b09d60d8beb9c1cb28b43066499940b969b
CUROBO_COMMIT=d17b54ce32cba095c0b000c4c58777075d11de0e

mkdir -p "${STATE_DIR}"

exec 9>"${STATE_DIR}/submodule_sparse.lock"
if ! flock -n 9; then
  echo "Another RoboDojo sparse submodule setup is already running" >&2
  exit 1
fi

fetch_commit() {
  local repo="$1"
  local commit="$2"
  local attempt

  if git -C "${repo}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    return 0
  fi

  for attempt in $(seq 1 20); do
    echo "[submodule] fetch ${commit} attempt ${attempt}"
    if git -C "${repo}" fetch \
      --depth 1 \
      --filter=blob:none \
      origin "${commit}" && \
      git -C "${repo}" cat-file -e "${commit}^{commit}"; then
      return 0
    fi
    sleep $((attempt * 3))
  done

  echo "Failed to fetch commit ${commit} in ${repo}" >&2
  return 1
}

set_head_and_sparse() {
  local repo="$1"
  local commit="$2"
  shift 2
  local head_ref
  local attempt

  git -C "${repo}" sparse-checkout init --cone
  git -C "${repo}" read-tree --reset "${commit}"
  head_ref=$(git -C "${repo}" symbolic-ref -q HEAD || true)
  if [[ -n "${head_ref}" ]]; then
    git -C "${repo}" update-ref "${head_ref}" "${commit}"
  else
    git -C "${repo}" update-ref HEAD "${commit}"
  fi

  for attempt in $(seq 1 20); do
    echo "[submodule] sparse checkout $(basename "${repo}") attempt ${attempt}"
    if git -C "${repo}" sparse-checkout set "$@"; then
      return 0
    fi
    sleep $((attempt * 3))
  done

  echo "Failed sparse checkout in ${repo}" >&2
  return 1
}

fetch_commit "${ISAACLAB_DIR}" "${ISAACLAB_COMMIT}"
set_head_and_sparse "${ISAACLAB_DIR}" "${ISAACLAB_COMMIT}" source scripts

fetch_commit "${CUROBO_DIR}" "${CUROBO_COMMIT}"
set_head_and_sparse "${CUROBO_DIR}" "${CUROBO_COMMIT}" src

for required in \
  "${ISAACLAB_DIR}/isaaclab.sh" \
  "${ISAACLAB_DIR}/source/isaaclab/setup.py" \
  "${CUROBO_DIR}/setup.py" \
  "${CUROBO_DIR}/src/curobo/__init__.py"; do
  [[ -f "${required}" ]] || {
    echo "Required submodule file is missing: ${required}" >&2
    exit 1
  }
done

git -C "${ISAACLAB_DIR}" diff --quiet
git -C "${ISAACLAB_DIR}" diff --cached --quiet
git -C "${CUROBO_DIR}" diff --quiet
git -C "${CUROBO_DIR}" diff --cached --quiet

printf 'IsaacLab=%s\nCuRobo=%s\nXPolicyLab=%s\n' \
  "$(git -C "${ISAACLAB_DIR}" rev-parse HEAD)" \
  "$(git -C "${CUROBO_DIR}" rev-parse HEAD)" \
  "$(git -C "${ROBODOJO_ROOT}/XPolicyLab" rev-parse HEAD)" \
  >"${STATE_DIR}/source_commits.txt"

printf 'complete\n' >"${STATE_DIR}/submodule_sparse.complete"
