#!/usr/bin/env bash
# Self-resubmitting auto-resume chain for the action-router (flow-matching head).
# LINEAGE-SCOPED: only considers run dirs whose timestamp >= CUTOFF, so the OLD
# MSE-head run (2026-08-23_22-17-51, has its own step_041940) and old smokes are
# ignored. Each job: (1) pick latest COMPLETE state ckpt in the lineage, (2) queue
# a follow-up that runs only if THIS job fails, (3) resume training. Stops when the
# lineage has weights/step_041940.pt, on a stop-file, or after MAX_ATTEMPTS.
# Manual stop: `touch .autoresume_stop` then scancel the running+pending jobs.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho020,gnho009,gnho017,gnho036,gnho011,gnho032
#SBATCH -J ap_router_auto
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail

REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
RUNGLOB="runs/robotwin_flux2_klein_4b_actionrouter_sub20"
CUTOFF="2026-08-24_21-04-42"     # first flow-matching real run; excludes old MSE + smokes
FINAL_STEP=041940
SELF="exploration/action_img_patch/sbatch_actionrouter_autoresume.sh"
STOP_FILE="${REPO_ROOT}/exploration/action_img_patch/.autoresume_stop"
CNT_FILE="${REPO_ROOT}/exploration/action_img_patch/.autoresume_count"
MAX_ATTEMPTS=15
SBATCH=/soft/slurm/bin/sbatch
SCANCEL=/soft/slurm/bin/scancel

lineage_run_dirs() {
  for r in $(ls -d ${RUNGLOB}/*/ 2>/dev/null); do
    b=$(basename "${r}")
    if [[ "${b}" > "${CUTOFF}" || "${b}" == "${CUTOFF}" ]]; then echo "${r%/}"; fi
  done
}

final_exists() {
  for r in $(lineage_run_dirs); do
    if [ -f "${r}/checkpoints/weights/step_${FINAL_STEP}.pt" ]; then return 0; fi
  done
  return 1
}

# ---- guards ------------------------------------------------------------
if [ -f "${STOP_FILE}" ]; then echo "[auto] stop-file present -> exit, no chain"; exit 0; fi
if final_exists; then echo "[auto] lineage final step ${FINAL_STEP} present -> done, no chain"; exit 0; fi
n=$(cat "${CNT_FILE}" 2>/dev/null || echo 0); n=$((n+1)); echo "${n}" > "${CNT_FILE}"
if [ "${n}" -gt "${MAX_ATTEMPTS}" ]; then echo "[auto] MAX_ATTEMPTS ${MAX_ATTEMPTS} reached -> stop"; exit 0; fi
echo "[auto] attempt ${n}/${MAX_ATTEMPTS} | cutoff=${CUTOFF}"

# ---- pick latest COMPLETE state ckpt within the lineage ----------------
RESUME_DIR=""
CAND=""
for r in $(lineage_run_dirs); do
  for s in $(ls -d "${r}/checkpoints/state/step_"* 2>/dev/null); do CAND="${CAND} ${s}"; done
done
for d in $(echo ${CAND} | tr " " "\n" | awk -F"step_" "NF>1{print \$2\" \"\$0}" | sort -rn | cut -d" " -f2); do
  if [ -d "${d}/pytorch_model" ] && [ -f "${d}/trainer_state.json" ]; then RESUME_DIR="${d}"; break; fi
done
if [ -z "${RESUME_DIR}" ]; then echo "[auto] no valid lineage state ckpt -> abort, no chain"; exit 0; fi
echo "[auto] resume from: ${RESUME_DIR}"

# ---- queue the follow-up (runs only if THIS job fails), spaced 3 min ----
NEXT=$(${SBATCH} --parsable --dependency=afternotok:${SLURM_JOB_ID} --begin=now+3minute "${SELF}" || true)
echo "[auto] queued follow-up job=${NEXT} (afternotok:${SLURM_JOB_ID})"

# ---- train -------------------------------------------------------------
source "${REPO_ROOT}/.venv/bin/activate"
export NCCL_SOCKET_IFNAME=eno1 GLOO_SOCKET_IFNAME=eno1 NCCL_IB_DISABLE=0
export HF_HUB_OFFLINE=1 WANDB_MODE=offline PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1 FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"
export MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29595}"
echo "[auto] node=${SLURM_JOB_NODELIST}"

/soft/slurm/bin/srun --export=ALL --kill-on-bad-exit=1 bash -c "
  export NNODES=1 NODE_RANK=0
  export PYTHONPATH=${REPO_ROOT}:\${PYTHONPATH:-}
  export TASK=robotwin_flux2_klein_4b_actionrouter_sub20
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 wandb.mode=offline resume=${RESUME_DIR}
"
RC=$?
echo "[auto] training exit code = ${RC}"

# ---- if final reached, cancel the pending follow-up --------------------
if final_exists; then
  echo "[auto] final reached -> cancel follow-up ${NEXT}"; [ -n "${NEXT}" ] && ${SCANCEL} "${NEXT}" 2>/dev/null || true
  exit 0
fi
exit ${RC}
