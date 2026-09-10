#!/usr/bin/env bash
set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state

mkdir -p "${STATE_DIR}"

exec 9>"${STATE_DIR}/sim_finish.lock"
if ! flock -n 9; then
  echo "Another RoboDojo sim finish process is already running" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DEFAULT_TIMEOUT=300
export TERM=xterm-256color

wait_for_job_marker() {
  local marker="$1"
  local pid_file="$2"
  local label="$3"
  local checks=0
  local pid

  while [[ ! -f "${marker}" ]]; do
    checks=$((checks + 1))
    if [[ "${checks}" -gt 1440 ]]; then
      echo "Timed out waiting for ${label}: ${marker}" >&2
      return 1
    fi

    pid=$(cat "${pid_file}" 2>/dev/null || true)
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
      echo "${label} stopped before writing ${marker}" >&2
      return 1
    fi

    echo "[sim-finish] waiting for ${label}; pid=${pid}"
    sleep 30
  done
}

wait_for_job_marker \
  "${STATE_DIR}/isaacsim_install.complete" \
  "${STATE_DIR}/isaacsim_install.pid" \
  "Isaac Sim install"

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate RoboDojo

cd "${ROBODOJO_ROOT}"

# Official RoboDojo installer entry: installs IsaacLab and CuRobo from the
# exact submodule commits already checked out under third_party/.
bash scripts/install.sh --from isaaclab

# demo_policy is code-only and lets the sim smoke test exercise client/server
# plumbing without downloading any policy checkpoint.
python -m pip install -e XPolicyLab

python - <<'PY'
import isaacsim  # noqa: F401
import isaaclab  # noqa: F401
import curobo  # noqa: F401

print("Isaac Sim, Isaac Lab, and CuRobo imports passed")
PY

printf 'complete\n' >"${STATE_DIR}/sim_packages.complete"

wait_for_job_marker \
  "${STATE_DIR}/asset_download.complete" \
  "${STATE_DIR}/asset_lfs_download.pid" \
  "RoboDojo asset download"

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

python -m pip freeze >"${STATE_DIR}/sim_pip_freeze.txt"
conda env export --no-builds >"${STATE_DIR}/sim_conda_env.yml"
printf 'complete\n' >"${STATE_DIR}/sim_setup.complete"
