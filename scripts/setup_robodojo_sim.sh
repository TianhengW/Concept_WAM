#!/usr/bin/env bash
set -euo pipefail

IMAGEWAM_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda
STATE_DIR=/storage/yukaichengLab/mazijian/wth/ImageWAM/data/.robodojo_state

mkdir -p "${STATE_DIR}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PIP_DEFAULT_TIMEOUT=300
export TERM=xterm-256color
export GIT_LFS_SKIP_SMUDGE=1

eval "$("${CONDA_BIN}" shell.bash hook)"

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx RoboDojo; then
  conda create -n RoboDojo -y \
    python=3.11 \
    cmake \
    ninja \
    ffmpeg \
    pkg-config
fi

conda activate RoboDojo

cd "${ROBODOJO_ROOT}"

git submodule sync third_party/IsaacLab third_party/curobo
git submodule update --init --depth 1 --filter=blob:none \
  third_party/IsaacLab \
  third_party/curobo

bash scripts/install.sh --from base_deps

HF_ENDPOINT=https://hf-mirror.com \
HF_REPO_ID=RoboDojo-Benchmark/RoboDojo \
HF_REPO_URL=https://hf-mirror.com/datasets/RoboDojo-Benchmark/RoboDojo \
bash scripts/init_assets.sh

python utils/update_embodiment_config_path.py

bash scripts/robodojo.sh doctor \
  --skip-policy \
  --summary "${STATE_DIR}/doctor.json"

python - <<'PY'
import isaaclab  # noqa: F401
import isaacsim  # noqa: F401
print("Isaac Sim and Isaac Lab imports passed")
PY

printf 'complete\n' > "${STATE_DIR}/sim_setup.complete"
