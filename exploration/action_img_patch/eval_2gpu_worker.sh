#!/bin/bash
# per-node worker: reads SEED/GLIST/CKPT/COMMON/RESDIR/LOGDIR from env, dispatches
# this node's slice of the 100-group list across its 8 GPUs. Skips groups that
# already have a full 100ep result in LOGDIR (resume-friendly).
set -uo pipefail
NODE="${SLURM_NODEID:-0}"
cd "${REPO_ROOT}"
source "${EVAL_ENV_ROOT}/.venv/bin/activate"
export HF_HUB_OFFLINE=1
export PATH="${EVAL_ENV_ROOT}/.local_bin:${PATH}"
export ROBOTWIN_EPISODE_TIMEOUT_S=300
export LD_LIBRARY_PATH="/usr/local/cuda-12.5/lib64:${EVAL_ENV_ROOT}/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

is_done() {
  local log=$1
  [ -f "$log" ] || return 1
  sed "s/\x1b\[[0-9;]*m//g" "$log" | grep -qE "Success rate: [0-9]+/10"
}

run_one() {
  local task=$1 cfg=$2 gpu=$3
  timeout -s KILL 9000 python experiments/robotwin/eval_robotwin_single.py \
    ckpt="${CKPT}" gpu_id=${gpu} seed=${SEED} \
    EVALUATION.task_name=${task} EVALUATION.task_config=${cfg} \
    EVALUATION.output_dir="${RESDIR}/${task}_${cfg}" ${COMMON} \
    > "${LOGDIR}/${task}_${cfg}.log" 2>&1 || echo "[node${NODE} gpu${gpu}] FAIL ${task} ${cfg}"
}

declare -a Q
for g in 0 1; do Q[$g]=""; done
i=0
for grp in ${GLIST}; do
  slot=$(( i % 2 ))
  if true; then
    g=$(( slot % 2 ))
    Q[$g]="${Q[$g]} ${grp}"
  fi
  i=$(( i + 1 ))
done

for g in 0 1; do
  (
    for grp in ${Q[$g]}; do
      task=${grp%%:*}; cfg=${grp##*:}
      LOG="${LOGDIR}/${task}_${cfg}.log"
      if is_done "$LOG"; then
        echo "[node${NODE} gpu${g}] SKIP(done) ${task} ${cfg}"
        continue
      fi
      echo "[node${NODE} gpu${g}] start ${task} ${cfg}"
      run_one "${task}" "${cfg}" "${g}"
      echo "[node${NODE} gpu${g}] done ${task} ${cfg}"
    done
  ) &
done
wait
echo "[node${NODE}] ALL DONE seed=${SEED}"
