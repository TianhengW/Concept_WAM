#!/usr/bin/env bash
set -euo pipefail

IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda
ARCHIVE_ROOT=/storage/yukaichengLab/share/datasets/.robodojo_source_archives
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
ISAACLAB_ARCHIVE=${ARCHIVE_ROOT}/IsaacLab-afca7b0.tar.gz
CUROBO_ARCHIVE=${ARCHIVE_ROOT}/curobo-d17b54c.tar.gz
ISAACLAB_DIR=${ROBODOJO_ROOT}/third_party/IsaacLab
CUROBO_DIR=${ROBODOJO_ROOT}/third_party/curobo
ISAACLAB_COMMIT=afca7b09d60d8beb9c1cb28b43066499940b969b
CUROBO_COMMIT=d17b54ce32cba095c0b000c4c58777075d11de0e

mkdir -p "${STATE_DIR}"

exec 9>"${STATE_DIR}/sim_setup.lock"
if ! flock -n 9; then
  echo "Another RoboDojo sim setup is already running" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DEFAULT_TIMEOUT=300
export TERM=xterm-256color

for archive in "${ISAACLAB_ARCHIVE}" "${CUROBO_ARCHIVE}"; do
  [[ -f "${archive}" ]] || {
    echo "Missing source archive: ${archive}" >&2
    exit 1
  }
  gzip -t "${archive}"
done

mkdir -p "${ISAACLAB_DIR}" "${CUROBO_DIR}"

tar -xzf "${ISAACLAB_ARCHIVE}" \
  -C "${ISAACLAB_DIR}" \
  --strip-components=1
tar -xzf "${CUROBO_ARCHIVE}" \
  -C "${CUROBO_DIR}" \
  --strip-components=1

git -C "${ISAACLAB_DIR}" read-tree --reset "${ISAACLAB_COMMIT}"
git -C "${ISAACLAB_DIR}" update-ref refs/heads/main "${ISAACLAB_COMMIT}"
git -C "${CUROBO_DIR}" read-tree --reset "${CUROBO_COMMIT}"
git -C "${CUROBO_DIR}" update-ref refs/heads/main "${CUROBO_COMMIT}"

git -C "${ISAACLAB_DIR}" diff --quiet
git -C "${ISAACLAB_DIR}" diff --cached --quiet
git -C "${CUROBO_DIR}" diff --quiet
git -C "${CUROBO_DIR}" diff --cached --quiet

printf 'IsaacLab=%s\nCuRobo=%s\nXPolicyLab=%s\n' \
  "$(git -C "${ISAACLAB_DIR}" rev-parse HEAD)" \
  "$(git -C "${CUROBO_DIR}" rev-parse HEAD)" \
  "$(git -C "${ROBODOJO_ROOT}/XPolicyLab" rev-parse HEAD)" \
  >"${STATE_DIR}/source_commits.txt"

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate RoboDojo

cd "${ROBODOJO_ROOT}"

python -m pip install -r scripts/requirements.txt
python -m pip install \
  opencv-python-headless==4.11.0.86 \
  pillow \
  matplotlib \
  scipy==1.15.3 \
  scikit-learn \
  numpy==1.26.0

bash scripts/install.sh --from isaacsim

python -m pip install -e XPolicyLab

if [[ ! -f "${STATE_DIR}/asset_download.complete" ]]; then
  echo "RoboDojo sim assets are not complete yet: ${STATE_DIR}/asset_download.complete" >&2
  exit 1
fi

for component in Robots Object Material Eval_Layout; do
  [[ -d "${ROBODOJO_ROOT}/Assets/${component}" ]] || {
    echo "Missing sim asset component: ${ROBODOJO_ROOT}/Assets/${component}" >&2
    exit 1
  }
done

python utils/update_embodiment_config_path.py

bash scripts/robodojo.sh doctor \
  --skip-policy \
  --summary "${STATE_DIR}/doctor.json"

python - <<'PY'
import isaacsim  # noqa: F401
import isaaclab  # noqa: F401
import curobo  # noqa: F401

print("Isaac Sim, Isaac Lab, and CuRobo imports passed")
PY

python -m pip freeze >"${STATE_DIR}/sim_pip_freeze.txt"
conda env export --no-builds >"${STATE_DIR}/sim_conda_env.yml"
printf 'complete\n' >"${STATE_DIR}/sim_setup.complete"
