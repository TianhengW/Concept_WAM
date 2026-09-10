#!/usr/bin/env bash
#SBATCH --job-name=robodojo_eval_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --exclude=nv-h100-029,nv-h100-007,gnho006,gnho041
#SBATCH --output=/storage/yukaichengLab/share/datasets/.robodojo_state/eval_smoke_%j.out
#SBATCH --error=/storage/yukaichengLab/share/datasets/.robodojo_state/eval_smoke_%j.out

# End-to-end RoboDojo sim eval smoke for the ImageWAM action-as-patch adapter:
# policy server (ImageWAM .venv, GPU 0) + Isaac Sim client (RoboDojo conda env, GPU 0),
# one episode of one task. Required env: IMAGEWAM_CKPT=/…/step_XXXXXX.pt
# Optional: IMAGEWAM_STATS, SMOKE_TASK (default stack_bowls), EVAL_NUM (default 1), CKPT_TAG.
set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
IMAGEWAM_VENV=/storage/yukaichengLab/mazijian/wth/ImageWAM/.venv
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
TASK=${SMOKE_TASK:-stack_bowls}
EVAL_NUM=${EVAL_NUM:-1}
CKPT_TAG=${CKPT_TAG:-imagewam_smoke}

[[ -f "${IMAGEWAM_CKPT:-}" ]] || { echo "IMAGEWAM_CKPT must point to a .pt file (got '${IMAGEWAM_CKPT:-}')" >&2; exit 1; }
export IMAGEWAM_CKPT
[[ -n "${IMAGEWAM_STATS:-}" ]] && export IMAGEWAM_STATS

echo "[eval-smoke] host=$(hostname) job=${SLURM_JOB_ID} task=${TASK} ckpt=${IMAGEWAM_CKPT} stats=${IMAGEWAM_STATS:-auto} $(date)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
# Isaac Sim 5.1 (Kit 107.3) dies with VkResult ERROR_DEVICE_LOST on the nodes that
# were upgraded to NVIDIA driver 610.x (gnho006, gnho041 seen so far); 555.x works.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d " ")
if [[ "${DRV%%.*}" -ge 600 ]]; then
  echo "[eval-smoke] DRIVER_MISMATCH host=$(hostname) driver=${DRV}: Kit/Vulkan crashes on 6xx drivers; resubmit with this node excluded" >&2
  exit 3
fi
echo "[eval-smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export PIP_USER=0
export TERM=xterm-256color
export HF_HUB_OFFLINE=1

# conda on PATH for the sim client scripts (`conda info --base`); policy side uses the venv.
eval "$(/storage/yukaichengLab/mazijian/miniconda3/bin/conda shell.bash hook)"
conda activate RoboDojo

cd "${ROBODOJO_ROOT}"
echo "[eval-smoke] launching robodojo eval $(date)"
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/imagewam_policy \
  --task "${TASK}" \
  --ckpt "${CKPT_TAG}" \
  --policy-env "${IMAGEWAM_VENV}" \
  --eval-env RoboDojo \
  --action-type joint \
  --seed 0 \
  --policy-gpu 0 \
  --env-gpu 0 \
  --eval-num "${EVAL_NUM}"
status=$?
echo "[eval-smoke] finished status=${status} $(date)"
find eval_result/RoboDojo/${TASK}/imagewam_policy -name "_result.json" -newer "${STATE_DIR}/eval_smoke_${SLURM_JOB_ID}.out" -exec sh -c 'echo "=== {}"; cat {}' \; 2>/dev/null || true
exit "${status}"
