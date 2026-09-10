#!/usr/bin/env bash
#SBATCH --job-name=robodojo-sim-smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/storage/yukaichengLab/share/datasets/.robodojo_state/sim_smoke_%j.out
#SBATCH --error=/storage/yukaichengLab/share/datasets/.robodojo_state/sim_smoke_%j.err

set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
CONDA_BIN=/storage/yukaichengLab/mazijian/miniconda3/bin/conda

mkdir -p "${STATE_DIR}"

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export TERM=xterm-256color

for vulkan_icd in \
  /usr/share/vulkan/icd.d/nvidia_icd.json \
  /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json \
  /etc/vulkan/icd.d/nvidia_icd.json; do
  if [[ -f "${vulkan_icd}" ]]; then
    export VK_ICD_FILENAMES="${vulkan_icd}"
    break
  fi
done

if [[ -z "${VK_ICD_FILENAMES:-}" ]]; then
  echo "NVIDIA Vulkan ICD was not found" >&2
  exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
conda activate RoboDojo

cd "${ROBODOJO_ROOT}"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python - <<'PY'
import isaaclab  # noqa: F401
import isaacsim  # noqa: F401
print("Isaac imports passed on GPU node")
PY

bash XPolicyLab/policy/demo_policy/install.sh

bash scripts/robodojo.sh doctor \
  --policy-dir XPolicyLab/policy/demo_policy \
  --sim-env RoboDojo \
  --policy-env RoboDojo \
  --summary "${STATE_DIR}/doctor_gpu.json"

timeout 2h bash scripts/robodojo.sh smoke \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo \
  --policy-env RoboDojo \
  --eval-env RoboDojo \
  --only stack_bowls \
  --eval-num 1 \
  --env-cfg arx_x5 \
  --action-type joint \
  --gpu-ids 0 \
  --fail-fast \
  --run-id sim_env_smoke \
  --summary "${STATE_DIR}/sim_smoke_summary.json" \
  --markdown "${STATE_DIR}/sim_smoke_summary.md"

printf 'complete\n' > "${STATE_DIR}/sim_smoke.complete"
