#!/usr/bin/env bash
#SBATCH --job-name=robodojo_smoke
#SBATCH --partition=yukaichenglab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=nv-h100-029,nv-h100-007,gnho006
#SBATCH --output=/storage/yukaichengLab/share/datasets/.robodojo_state/smoke_%j.out
#SBATCH --error=/storage/yukaichengLab/share/datasets/.robodojo_state/smoke_%j.out

# RoboDojo simulator smoke test on H800: demo_policy (zero actions, no
# checkpoint) drives one episode of one task end-to-end through the real
# policy-server/env-client wiring and Isaac Sim headless rendering.
set -euo pipefail

ROBODOJO_ROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM/third_party/RoboDojo
STATE_DIR=/storage/yukaichengLab/share/datasets/.robodojo_state
TASK=${SMOKE_TASK:-stack_bowls}
RUN_ID="smoke_demo_${SLURM_JOB_ID}"

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID} task=${TASK} $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[smoke] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Wait (up to 45 min) for the simulator install to finish if it is still running.
for _ in $(seq 1 90); do
  [[ -f "${STATE_DIR}/sim_setup.complete" ]] && break
  echo "[smoke] waiting for ${STATE_DIR}/sim_setup.complete ... $(date +%H:%M:%S)"
  sleep 30
done
[[ -f "${STATE_DIR}/sim_setup.complete" ]] || { echo "[smoke] install marker never appeared" >&2; exit 1; }

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONNOUSERSITE=1
export PIP_USER=0
export TERM=xterm-256color

eval "$(/storage/yukaichengLab/mazijian/miniconda3/bin/conda shell.bash hook)"
conda activate RoboDojo
echo "[smoke] python=$(which python) LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
python -c "import torch; print('[smoke] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd "${ROBODOJO_ROOT}"
bash scripts/robodojo.sh doctor --skip-policy --summary "${STATE_DIR}/doctor_${SLURM_JOB_ID}.json"

echo "[smoke] launching robodojo smoke $(date)"
bash scripts/robodojo.sh smoke \
  --only "${TASK}" \
  --policy-dir XPolicyLab/policy/demo_policy \
  --ckpt demo \
  --policy-env demo-policy \
  --eval-env RoboDojo \
  --action-type joint \
  --seed 0 \
  --policy-gpu 0 \
  --env-gpu 0 \
  --run-id "${RUN_ID}" \
  --summary "${STATE_DIR}/${RUN_ID}.json" \
  --markdown "${STATE_DIR}/${RUN_ID}.md"
status=$?
echo "[smoke] finished status=${status} $(date)"
echo "=== summary json ==="
cat "${STATE_DIR}/${RUN_ID}.json" 2>/dev/null || true
exit "${status}"
