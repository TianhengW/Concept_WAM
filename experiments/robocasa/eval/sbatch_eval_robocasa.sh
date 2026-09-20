#!/usr/bin/env bash
# Evaluate an Action-as-Patch checkpoint on RoboCasa365: N GPUs on one node -> N server/client pairs,
# tasks round-robin over workers, then merged summary.json.
#   CKPT=<weights .pt> RUN_DIR=<train run dir> [TASK_SET=target50] [TASKS="OpenDrawer CloseFridge"] [NUM_TRIALS=50]
#   [SPLIT=pretrain] [REPLAN=16] [EXTRA="--save-failure-videos"] [RUN_ID=...] \
#   sbatch --gres=gpu:8 experiments/robocasa/eval/sbatch_eval_robocasa.sh
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -J ap_rc365_eval
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail
export PATH=/soft/slurm/bin:${PATH}
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa"
cd "${REPO_ROOT}"
CKPT="${CKPT:?}"; RUN_DIR="${RUN_DIR:?}"
TASK_SET="${TASK_SET:-target50}"; TASKS="${TASKS:-}"; NUM_TRIALS="${NUM_TRIALS:-50}"; SPLIT="${SPLIT:-pretrain}"; REPLAN="${REPLAN:-16}"
EXTRA="${EXTRA:-}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/eval_results/robocasa365}"
RUN_ID="${RUN_ID:-$(basename "$(dirname "$(dirname "$(dirname "${CKPT}")")")")_$(basename "${CKPT}" .pt)_${SLURM_JOB_ID}}"
NGPU=$(nvidia-smi -L | wc -l)
# Snapshot the eval scripts into the run dir: bash reads scripts incrementally, so editing the repo copy while a job
# runs would corrupt it; the snapshot also records exactly which code produced the results.
CODE_DIR="${OUT_ROOT}/${RUN_ID}/code"; mkdir -p "${CODE_DIR}"
cp experiments/robocasa/eval/{run_eval_pair.sh,ap_policy_server.py,eval_robocasa_ap.py,merge_summaries.py} "${CODE_DIR}/"
# The NVIDIA EGL driver on the gnho nodes (libnvidia-eglcore 610.43.02) aborts inside mjr_readPixels as soon as
# a second process (a CUDA policy server OR another EGL client) shares the GPU with a rendering sim client.
# So every client gets a render GPU of its own: the last N_RENDER GPUs render only, the first N_PAIRS host the
# policy servers, one client per render GPU. Default N_RENDER = NGPU/2 (8 GPUs -> 4 pairs + 4 render GPUs).
# ALLOW_SHARED_RENDER=1 lets clients share render GPUs (crashes on 610 drivers; maybe fine elsewhere).
# RENDER_BACKEND=osmesa: clients render on CPU (~6x slower steps, no driver aborts) -> every GPU hosts a policy pair.
RENDER_BACKEND="${RENDER_BACKEND:-egl}"; export RENDER_BACKEND
if [ "${RENDER_BACKEND}" = "osmesa" ]; then
  N_RENDER=0; N_PAIRS=${NGPU}
else
  N_RENDER="${N_RENDER:-$((NGPU / 2))}"
  if [ "${N_RENDER}" -ge "${NGPU}" ] || [ "${N_RENDER}" -lt 1 ]; then echo "N_RENDER=${N_RENDER} must be in [1, gpus-1]; gpus=${NGPU}"; exit 2; fi
  N_PAIRS=$((NGPU - N_RENDER))
  if [ "${N_PAIRS}" -gt "${N_RENDER}" ] && [ "${ALLOW_SHARED_RENDER:-0}" != "1" ]; then
    echo "[eval] ${N_PAIRS} pairs > ${N_RENDER} render GPUs: clients would share a render GPU (EGL aborts on this cluster). Set N_RENDER>=pairs, ALLOW_SHARED_RENDER=1, or RENDER_BACKEND=osmesa."; exit 2
  fi
fi
TASK_ARGS=(); for t in ${TASKS}; do TASK_ARGS+=(--task-name "${t}"); done
echo "[eval] job=${SLURM_JOB_ID} node=$(hostname) gpus=${NGPU} policy_pairs=${N_PAIRS} render_gpus=${N_RENDER} backend=${RENDER_BACKEND} ckpt=${CKPT} task_set=${TASK_SET} tasks=[${TASKS:-all}] trials=${NUM_TRIALS} split=${SPLIT} replan=${REPLAN} -> ${OUT_ROOT}/${RUN_ID}"
# per-job port range so several eval jobs can share a node without colliding
BASE_PORT="${BASE_PORT:-$((20000 + (SLURM_JOB_ID % 500) * 16))}"
PIDS=()
for i in $(seq 0 $((N_PAIRS-1))); do
  if [ "${N_RENDER}" -gt 0 ]; then EGL=$((N_PAIRS + i % N_RENDER)); else EGL=$i; fi
  GPU=$i EGL_GPU=$EGL PORT=$((BASE_PORT+i)) CKPT="${CKPT}" RUN_DIR="${RUN_DIR}" OUT_ROOT="${OUT_ROOT}" RUN_ID="${RUN_ID}" WORKER_INDEX=$i NUM_WORKERS=${N_PAIRS} \
    bash "${CODE_DIR}/run_eval_pair.sh" --task-set "${TASK_SET}" --num-trials "${NUM_TRIALS}" --split "${SPLIT}" --replan-steps "${REPLAN}" \
    ${TASK_ARGS[@]+"${TASK_ARGS[@]}"} ${EXTRA} &
  PIDS+=($!)
done
FAIL=0; for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
python3 "${CODE_DIR}/merge_summaries.py" "${OUT_ROOT}/${RUN_ID}"
echo "[eval] DONE fail=${FAIL} results=${OUT_ROOT}/${RUN_ID}"
exit ${FAIL}
