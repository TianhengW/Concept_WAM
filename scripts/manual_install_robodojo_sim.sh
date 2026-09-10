#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground-only RoboDojo simulator environment installer for H800.
# It installs the evaluation/simulator stack and required simulator assets.
# It does not download policy checkpoints, real-robot assets, depth data, or
# training data.

IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
ROBODOJO_ROOT=${IMAGEWAM_ROOT}/third_party/RoboDojo
CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
ASSET_SCRIPT=${IMAGEWAM_ROOT}/scripts/download_robodojo_assets_lfs_share.sh

ENV_NAME=RoboDojo
INSTALL_ASSETS=1

usage() {
  cat <<'EOF'
Usage:
  bash manual_install_robodojo_sim.sh [--skip-assets]

Options:
  --skip-assets  Install Python/Isaac packages only. The default also downloads
                 the four simulator asset trees required for evaluation.
  -h, --help     Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-assets)
      INSTALL_ASSETS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/manual_sim_install.lock"
if ! flock -n 9; then
  echo "Another manual RoboDojo simulator install is already running." >&2
  exit 1
fi

exec > >(tee -a "${STATE_DIR}/manual_sim_install.log") 2>&1

on_error() {
  local status=$?
  echo "[ERROR] Manual RoboDojo install failed with status ${status}."
  echo "[ERROR] Log: ${STATE_DIR}/manual_sim_install.log"
  exit "${status}"
}
trap on_error ERR

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DEFAULT_TIMEOUT=600
export PIP_RETRIES=50
export TERM=xterm-256color
export ISAACLAB_RL_FRAMEWORK=none

[[ -x "${CONDA_BIN}" ]] || {
  echo "Conda was not found at ${CONDA_BIN}" >&2
  exit 1
}
[[ -d "${ROBODOJO_ROOT}/.git" ]] || {
  echo "RoboDojo repo was not found at ${ROBODOJO_ROOT}" >&2
  exit 1
}
[[ -x "${ASSET_SCRIPT}" ]] || {
  echo "Asset downloader was not found or executable: ${ASSET_SCRIPT}" >&2
  exit 1
}

eval "$("${CONDA_BIN}" shell.bash hook)"

if ! "${CONDA_BIN}" env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[1/9] Creating ${ENV_NAME} with Python 3.11"
  "${CONDA_BIN}" create -n "${ENV_NAME}" python=3.11 -y
else
  echo "[1/9] Reusing existing ${ENV_NAME} environment"
fi

conda activate "${ENV_NAME}"
[[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]] || {
  echo "Failed to activate ${ENV_NAME}" >&2
  exit 1
}

echo "[2/9] Installing command-line build/video dependencies into Conda"
missing_conda_tools=()
command -v cmake >/dev/null 2>&1 || missing_conda_tools+=(cmake)
command -v ffmpeg >/dev/null 2>&1 || missing_conda_tools+=(ffmpeg)
if [[ ${#missing_conda_tools[@]} -gt 0 ]]; then
  "${CONDA_BIN}" install -n "${ENV_NAME}" -y -c conda-forge "${missing_conda_tools[@]}"
fi
command -v gcc >/dev/null 2>&1 || {
  echo "gcc is missing; ask the node administrator to install build-essential." >&2
  exit 1
}
command -v g++ >/dev/null 2>&1 || {
  echo "g++ is missing; ask the node administrator to install build-essential." >&2
  exit 1
}

echo "[3/9] Verifying locked RoboDojo submodules"
cd "${ROBODOJO_ROOT}"
declare -A EXPECTED_SUBMODULES=(
  [XPolicyLab]=432f82b1758c5b1202e42a3dfe014546dbc50871
  [third_party/IsaacLab]=afca7b09d60d8beb9c1cb28b43066499940b969b
  [third_party/curobo]=d17b54ce32cba095c0b000c4c58777075d11de0e
)
for submodule in XPolicyLab third_party/IsaacLab third_party/curobo; do
  [[ -d "${submodule}/.git" || -f "${submodule}/.git" ]] || {
    echo "Missing submodule: ${submodule}" >&2
    echo "Restore the prepared sparse submodule before continuing." >&2
    exit 1
  }
  actual_commit=$(git -C "${submodule}" rev-parse HEAD)
  expected_commit=${EXPECTED_SUBMODULES[${submodule}]}
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "Unexpected ${submodule} commit: ${actual_commit}" >&2
    echo "Expected: ${expected_commit}" >&2
    exit 1
  fi
done
git submodule status

echo "[4/9] Installing PyTorch 2.7/cu128 and Isaac Sim 5.1"
bash "${IMAGEWAM_ROOT}/scripts/install_robodojo_isaacsim_only.sh"
conda activate "${ENV_NAME}"

echo "[5/9] Installing RoboDojo runtime dependencies"
python -m pip install -r scripts/requirements.txt
python -m pip install \
  opencv-python-headless==4.11.0.86 \
  pillow \
  matplotlib \
  scipy==1.15.3 \
  scikit-learn \
  numpy==1.26.0

echo "[6/9] Running the official Isaac Lab and CuRobo installer"
bash scripts/install.sh --from isaaclab
conda activate "${ENV_NAME}"

# Do not install XPolicyLab into the simulator environment. XPolicyLab's own
# documentation keeps policy and simulator dependencies isolated, and this
# simulator-only setup intentionally downloads no policy package or checkpoint.

echo "[7/9] Verifying simulator package imports"
python - <<'PY'
import isaacsim
import isaaclab
import curobo

print("Isaac Sim, Isaac Lab, and CuRobo imports passed")
PY

# The official RoboDojo installer pins starlette==0.45.3 (isaacsim-core
# constraint) while isaaclab's metadata declares starlette==0.49.1.  That single
# conflict is expected; anything else reported by pip check is still fatal.
if ! pip_check_output=$(python -m pip check 2>&1); then
  echo "${pip_check_output}"
  unexpected_conflicts=$(grep -vE '^isaaclab [0-9.]+ has requirement starlette==' <<<"${pip_check_output}" || true)
  if [[ -n "${unexpected_conflicts}" ]]; then
    echo "Unexpected pip check conflicts:" >&2
    echo "${unexpected_conflicts}" >&2
    exit 1
  fi
  echo "[WARN] pip check: only the known starlette pin conflict (official installer 0.45.3 vs isaaclab 0.49.1); continuing."
fi
python -m pip show isaacsim isaaclab nvidia-curobo torch torchvision torchaudio \
  >"${STATE_DIR}/manual_sim_versions.txt"

if [[ "${INSTALL_ASSETS}" -eq 1 ]]; then
  echo "[8/9] Downloading only the required simulator assets"
  bash "${ASSET_SCRIPT}"
else
  echo "[8/9] Skipping simulator assets by request"
fi

echo "[9/9] Updating asset paths and running RoboDojo doctor"
if [[ "${INSTALL_ASSETS}" -eq 1 ]]; then
  for component in Robots Object Material Eval_Layout; do
    [[ -d "${ROBODOJO_ROOT}/Assets/${component}" ]] || {
      echo "Missing simulator asset component: Assets/${component}" >&2
      exit 1
    }
  done
  python utils/update_embodiment_config_path.py
fi

bash scripts/robodojo.sh doctor \
  --skip-policy \
  --summary "${STATE_DIR}/doctor.json"

python -m pip freeze >"${STATE_DIR}/sim_pip_freeze.txt"
conda env export --no-builds >"${STATE_DIR}/sim_conda_env.yml"
printf 'complete\n' >"${STATE_DIR}/sim_setup.complete"

echo
echo "RoboDojo simulator environment installation completed."
echo "Environment: ${ENV_NAME}"
echo "Repository:  ${ROBODOJO_ROOT}"
echo "Doctor:      ${STATE_DIR}/doctor.json"
echo "Next step:"
echo "  /soft/slurm/bin/sbatch ${IMAGEWAM_ROOT}/scripts/robodojo_sim_smoke.sbatch"
