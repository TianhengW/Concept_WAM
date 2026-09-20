#!/usr/bin/env bash
# Smoke + per-GPU batch sweep for action-as-patch x RoboCasa365 pretrain on nv-h100-007
# (8 GPUs, ZeRO-1). Reports peak GPU memory / mean SM util per batch so we can pick the
# largest batch that stays >= 80% memory without OOM. Later sweep iterations reuse the
# dataset_stats.json produced by the first one.
#   Usage: [BATCHES="8 12 16 20 24"] [STEPS=12] sbatch experiments/robocasa/sbatch_smoke_robocasa_007.sh
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH -w nv-h100-007
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH -J ap_rc365_smoke007
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail   # no -e: keep sweeping after a failed batch is reported

REPO_ROOT=/storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source /storage/yukaichengLab/mazijian/wth/ImageWAM/.venv/bin/activate

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="${REPO_ROOT}"
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export TASK=robocasa_flux2_klein_4b_actionpatch_pretrain

BATCHES="${BATCHES:-8 12 16 20 24}"
STEPS="${STEPS:-12}"
LOGDIR="${REPO_ROOT}/experiments/robocasa/logs/smoke_${SLURM_JOB_ID}"
mkdir -p "${LOGDIR}"

echo "=== node ==="; hostname
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "--- ifaces ---"; ip -o -4 addr show | awk '{print $2, $4}'
echo "--- ib ---"; (ibv_devinfo -l 2>/dev/null || ibstat 2>/dev/null | head -5 || echo "no ib tools")
TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "=== sweep batches=[${BATCHES}] steps=${STEPS} total_mem=${TOTAL_MIB}MiB target>=80% ==="

STATS=""
for B in ${BATCHES}; do
  RID="smoke_b${B}_${SLURM_JOB_ID}"
  MEM="${LOGDIR}/gpu_b${B}.csv"
  TLOG="${LOGDIR}/train_b${B}.log"
  EXTRA=()
  if [ -n "${STATS}" ]; then
    EXTRA+=("data.train.pretrained_norm_stats=${STATS}" "data.val.pretrained_norm_stats=${STATS}")
  fi
  ( while true; do nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits; sleep 2; done > "${MEM}" ) &
  MP=$!
  T0=$(date +%s)
  RUN_ID="${RID}" bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
      batch_size="${B}" max_steps="${STEPS}" log_every=1 num_workers=8 \
      save_every=1000000 eval_every=1000000 wandb.enabled=false \
      ${EXTRA[@]+"${EXTRA[@]}"} > "${TLOG}" 2>&1
  RC=$?
  T1=$(date +%s)
  kill "${MP}" 2>/dev/null; wait "${MP}" 2>/dev/null
  PEAK=$(cut -d, -f2 "${MEM}" | sort -n | tail -1)
  PEAK_PCT=$(python3 -c "print(f'{100*${PEAK:-0}/${TOTAL_MIB}:.1f}')")
  UTIL=$(awk -F', *' '$3>10{s+=$3;n++} END{if(n) printf "%.0f", s/n; else print 0}' "${MEM}")
  OOM=$(grep -c -i -E "out of memory|OutOfMemoryError" "${TLOG}")
  echo "[smoke] batch=${B} rc=${RC} elapsed=$((T1-T0))s peak_mem=${PEAK:-?}MiB (${PEAK_PCT}% of ${TOTAL_MIB}) mean_util=${UTIL}% oom=${OOM}"
  if [ -z "${STATS}" ] && [ -f "runs/${TASK}/${RID}/dataset_stats.json" ]; then
    STATS="${REPO_ROOT}/runs/${TASK}/${RID}/dataset_stats.json"
    echo "[smoke] dataset_stats -> ${STATS}"
  fi
  if [ "${OOM}" != "0" ] || [ "${RC}" != "0" ]; then
    echo "[smoke] batch=${B} FAILED (rc=${RC} oom=${OOM}) -> stop sweep; see ${TLOG}"
    tail -5 "${TLOG}"
    break
  fi
done
echo "[smoke] DONE"
