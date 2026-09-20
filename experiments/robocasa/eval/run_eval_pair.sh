#!/usr/bin/env bash
# One GPU = one AP policy server (AP venv) + one RoboCasa rollout client (robocasa conda env).
# Env vars: GPU PORT CKPT RUN_DIR [STATS] OUT_ROOT RUN_ID WORKER_INDEX NUM_WORKERS [NUM_INFERENCE_STEPS] ; extra client args as "$@"
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-/storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa}"
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python entrypoints live next to this script (snapshot-safe)
AP_VENV="${AP_VENV:-/storage/yukaichengLab/mazijian/wth/ImageWAM/.venv}"
ROBOCASA_PY="${ROBOCASA_PY:-/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/.conda-robocasa365/bin/python}"
MODALITY_JSON="${MODALITY_JSON:-/storage/yukaichengLab/share/datasets/robocasa365/pretrain/atomic/CloseDrawer/20250819/lerobot/meta/modality.json}"
GPU="${GPU:?}"; PORT="${PORT:?}"; CKPT="${CKPT:?}"; RUN_DIR="${RUN_DIR:?}"
EGL_GPU="${EGL_GPU:-${GPU}}"   # GPU used by the sim client for EGL rendering (may differ from the policy GPU)
OUT_ROOT="${OUT_ROOT:?}"; RUN_ID="${RUN_ID:?}"; WORKER_INDEX="${WORKER_INDEX:-0}"; NUM_WORKERS="${NUM_WORKERS:-1}"
STATS="${STATS:-${RUN_DIR}/dataset_stats.json}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

LOG_DIR="${OUT_ROOT}/${RUN_ID}/logs"; mkdir -p "${LOG_DIR}"
READY="${LOG_DIR}/server_${PORT}.ready"; rm -f "${READY}"
cd "${REPO_ROOT}"

# --- server (AP venv, GPU) ---
(
  export CUDA_VISIBLE_DEVICES="${GPU}"
  source "${AP_VENV}/bin/activate"
  set -a; source "${REPO_ROOT}/.env.local"; set +a
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  export REPO_ROOT
  exec python "${CODE_DIR}/ap_policy_server.py" --run-dir "${RUN_DIR}" --ckpt "${CKPT}" --stats "${STATS}" \
       --port "${PORT}" --device cuda:0 --num-inference-steps "${NUM_INFERENCE_STEPS}" --ready-file "${READY}"
) > "${LOG_DIR}/server_${PORT}.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 240); do
  [ -f "${READY}" ] && break
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then echo "[pair gpu=${GPU}] server died during startup; see ${LOG_DIR}/server_${PORT}.log"; tail -20 "${LOG_DIR}/server_${PORT}.log"; exit 1; fi
  sleep 5
done
[ -f "${READY}" ] || { echo "[pair gpu=${GPU}] server not ready after 20 min"; kill "${SERVER_PID}"; exit 1; }
echo "[pair gpu=${GPU}] server ready on port ${PORT} (client renders on gpu ${EGL_GPU})"

# --- client (robocasa conda env, headless EGL) ---
# mujoco/EGL occasionally aborts natively on env.reset(); the client resumes from <task>/stats.json,
# so restart it until it exits cleanly (give up after MAX_RESTARTS or 3 attempts without progress).
export PYTHONFAULTHANDLER=1
RENDER_BACKEND="${RENDER_BACKEND:-egl}"
if [ "${RENDER_BACKEND}" = "osmesa" ]; then
  # CPU software rendering (~0.2 s/step vs 0.035 on EGL) - immune to the EGL driver aborts, needs no render GPU
  OSMESA_LIB="${OSMESA_LIB:-/storage/yukaichengLab/mazijian/envs/osmesa_24.0.7/lib}"
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="${OSMESA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CUDA_VISIBLE_DEVICES=""; unset MUJOCO_EGL_DEVICE_ID
  export OMP_NUM_THREADS="${OSMESA_THREADS:-8}" MKL_NUM_THREADS=1
else
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export CUDA_VISIBLE_DEVICES="${EGL_GPU}" MUJOCO_EGL_DEVICE_ID="${EGL_GPU}"   # robosuite asserts EGL id is in CUDA_VISIBLE_DEVICES
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1                          # single-threaded sim loop (as in the reference evaluator)
fi
# The client runs --max-episodes-per-run episodes (default 1) and exits 3 while work remains, 0 when done;
# any other exit code is a crash: restart (the client resumes from <task>/stats.json + inflight.json).
MAX_CRASHES="${MAX_CRASHES:-200}"
CLIENT_LOG="${LOG_DIR}/client_${PORT}.log"
runs=0; crashes=0; noprog=0; RC=1
while :; do
  runs=$((runs+1))
  before=$(grep -c -E "EPISODE_END|SKIP task" "${CLIENT_LOG}" 2>/dev/null || true)
  # CLIENT_WRAPPER (debug): e.g. 'gdb -batch -ex run -ex "bt 40" --args' to capture the C stack of a native abort
  ${CLIENT_WRAPPER:-} "${ROBOCASA_PY}" "${CODE_DIR}/eval_robocasa_ap.py" --server-port "${PORT}" --modality-json "${MODALITY_JSON}" \
      --save-root-dir "${OUT_ROOT}" --run-id "${RUN_ID}" --worker-index "${WORKER_INDEX}" --num-workers "${NUM_WORKERS}" "$@" \
      >> "${CLIENT_LOG}" 2>&1
  RC=$?
  if [ -n "${CLIENT_WRAPPER:-}" ]; then echo "[pair gpu=${GPU}] debug wrapper run finished rc=${RC}; not restarting"; break; fi
  [ "${RC}" -eq 0 ] && break
  if [ "${RC}" -eq 3 ]; then noprog=0; continue; fi
  crashes=$((crashes+1))
  after=$(grep -c -E "EPISODE_END|SKIP task" "${CLIENT_LOG}" 2>/dev/null || true)
  if [ "${after}" -gt "${before:-0}" ]; then noprog=0; else noprog=$((noprog+1)); fi
  echo "[pair gpu=${GPU}] client crashed rc=${RC} (crash ${crashes}, run ${runs}, episodes so far ${after}, no-progress streak ${noprog})"
  if [ "${crashes}" -ge "${MAX_CRASHES}" ] || [ "${noprog}" -ge 6 ]; then echo "[pair gpu=${GPU}] giving up"; break; fi
  sleep 2
done
echo "[pair gpu=${GPU}] client exit=${RC} after ${runs} run(s), ${crashes} crash(es)"
kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null
exit "${RC}"
