#!/usr/bin/env bash
# accelerate launcher for v3 (VL LatentReasoner). Single- or multi-node via env:
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPU_PER_NODE.
# Hydra overrides pass through as extra args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env_v3.sh"

GPU_PER_NODE="${GPU_PER_NODE:-8}"
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29533}"
ZERO_STAGE="${ZERO_STAGE:-2}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
TASK_BASENAME="${TASK_BASENAME:-robotwin_v3}"

case "${ZERO_STAGE}" in
  1|zero1) ACCEL_CFG="${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero1_ds.yaml" ;;
  2|zero2) ACCEL_CFG="${REPO_ROOT}/scripts/accelerate_configs/accelerate_zero2_ds.yaml" ;;
  *) echo "Invalid ZERO_STAGE=${ZERO_STAGE}" >&2; exit 1 ;;
esac

cd "${REPO_ROOT}"

echo "[launch_v3] nodes=${NUM_MACHINES} rank=${MACHINE_RANK} master=${MAIN_PROCESS_IP}:${MAIN_PROCESS_PORT} gpus=${GPU_PER_NODE} zero=${ZERO_STAGE} run_id=${RUN_ID}"

MULTI_ARGS=()
if (( NUM_MACHINES > 1 )); then
  MULTI_ARGS=(
    --num_machines "${NUM_MACHINES}"
    --machine_rank "${MACHINE_RANK}"
    --main_process_ip "${MAIN_PROCESS_IP}"
    --main_process_port "${MAIN_PROCESS_PORT}"
    --deepspeed_multinode_launcher standard
  )
fi

accelerate launch \
  --config_file "${ACCEL_CFG}" \
  --num_processes "$(( GPU_PER_NODE * NUM_MACHINES ))" \
  --num_machines "${NUM_MACHINES}" \
  "${MULTI_ARGS[@]}" \
  "${V3_ROOT}/scripts/train_v3.py" \
  "output_dir=${V3_ROOT}/runs/${TASK_BASENAME}/${RUN_ID}" \
  "resume=${RESUME:-${BASE_CKPT}}" \
  "model.flux2_src_path=${FLUX2_SRC}" \
  "model.flux2_model_path=${FLUX2_MODEL_PATH}" \
  "model.ae_model_path=${FLUX2_AE_MODEL_PATH}" \
  "model.vl_model_path=${VL_MODEL_PATH}" \
  "data.train.dataset_dirs=[${DATA_DIR}]" \
  "data.val.dataset_dirs=[${DATA_DIR}]" \
  "wandb.name=${TASK_BASENAME}" \
  "$@"
