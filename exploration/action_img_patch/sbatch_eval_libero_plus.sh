#!/usr/bin/env bash
# LIBERO-plus (perturbation benchmark, arXiv 2510.13626) evaluation for action-as-patch ckpts on H800.
# Runtime mirrors xz/experiments/libero_plus_h800_600x1_20260821_r1/run_600_full.sh (validated 10x1 on 08-21):
# xz python env + LIBERO-plus source (support r3) + ImageMagick6 + LIBERO_CONFIG_PATH=xz config r2.
# The task manifest (suite,task_id per line) is split round-robin into NGPU*PER_GPU chunks, one worker each.
# Robustness (08-28 lesson: 10/12 workers SIGABRT'd during the concurrent start-up burst on gnho006, no traceback):
#   workers start staggered (STAGGER s apart), run with -X faulthandler, and up to ROUNDS rounds re-run only the
#   tasks that still lack <OUT>/<suite>/gpu0_task<id>_results.json (results are written per task, so nothing is lost).
# Env knobs: CKPT (required), STATS (default <run_dir>/dataset_stats.json), TASK (hydra task config),
#   MANIFEST (default xz 600-task seed20260819 manifest), NUM_TRIALS (1), NGPU (8), PER_GPU (2),
#   REPLAN (12), HORIZON (16), OUT_TAG, ROUNDS (3), STAGGER (10).
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -x gnho006,gnho020,gnho041
#SBATCH -J eval_libero_plus
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail
REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM
cd "$REPO"; mkdir -p slurm_logs
XZ=/storage/yukaichengLab/mazijian/xz
ENV=$XZ/runtime/libero_plus_h800_native_20260820_r1
SUP=$XZ/runtime/libero_plus_h800_native_support_20260821_r3
MAGICK=$XZ/runtime/imagemagick6_native_20260820_r2
PLUS="$SUP/source/LIBERO-plus"
FLUX2=/storage/yukaichengLab/mazijian/wth/flux2
FLUX_MODEL=/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
AE_MODEL=/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors
QWEN_MODEL=/storage/yukaichengLab/share/model/Qwen3-4B

CKPT="${CKPT:?set CKPT}"
# CKPT may end in "/latest": resolve at job start to the newest step_*.pt in that weights dir (so a job queued for
# days evaluates the final checkpoint of a still-running training run).
if [ "$(basename "$CKPT")" = "latest" ]; then
  CKPT=$(ls -1 "$(dirname "$CKPT")"/step_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)
  [ -n "$CKPT" ] || { echo "ERROR: no step_*.pt found for CKPT=latest"; exit 1; }
  echo "[ckpt] resolved latest -> $CKPT"
fi
RUN_DIR="$(dirname "$(dirname "$(dirname "$CKPT")")")"
STATS="${STATS:-$RUN_DIR/dataset_stats.json}"
TASK="${TASK:-libero_flux2_klein_4b_actionpatch}"
MANIFEST="${MANIFEST:-$XZ/experiments/libero_plus_h800_600x1_20260821_r1/task_600x1.csv}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NGPU="${NGPU:-8}"; PER_GPU="${PER_GPU:-2}"
REPLAN="${REPLAN:-12}"; HORIZON="${HORIZON:-16}"
CKPT_TAG="$(basename "$(dirname "$RUN_DIR")")_$(basename "$RUN_DIR")"
OUT_TAG="${OUT_TAG:-eval_${SLURM_JOB_ID}_$(basename "$CKPT" .pt)}"
OUT="$REPO/evaluate_results/libero_plus/$CKPT_TAG/$OUT_TAG"
mkdir -p "$OUT/logs" "$OUT/chunks"

export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUP/python-overlay:$SUP/transformers-overlay/transformers451-r1:$PLUS:$REPO:$REPO/src:$FLUX2/src:$FLUX2"
export LIBERO_CONFIG_PATH="$XZ/config/libero_plus_eval_20260821_r2"
export MAGICK_HOME="$MAGICK" MAGICK_CONFIGURE_PATH="$MAGICK/etc/ImageMagick-6"
export LD_LIBRARY_PATH="$MAGICK/lib:/storage/soft/cuda/cuda-12.5/lib64:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# RENDER=osmesa (default, 2026-08-29): CPU llvmpipe rendering via conda-forge mesalib 24.0.7 (libOSMesa.so.8). EGL on the
# H800 nodes is unreliable: GLVND mixes Mesa software devices into the EGL device list and workers SIGABRT in
# mjr_readPixels (see memory aap-libero-h800-campaign). RENDER=egl switches back. OMP_NUM_THREADS bounds llvmpipe threads.
RENDER="${RENDER:-osmesa}"
OSMESA_LIB=/storage/yukaichengLab/mazijian/envs/osmesa_24.0.7/lib
if [ "$RENDER" = "osmesa" ]; then
  export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LD_LIBRARY_PATH="$OSMESA_LIB:$LD_LIBRARY_PATH" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
else
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false GIT_PYTHON_REFRESH=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== START $(date) node=$SLURMD_NODENAME ckpt=$CKPT stats=$STATS task=$TASK manifest=$MANIFEST ($(wc -l < "$MANIFEST") rows) trials=$NUM_TRIALS out=$OUT"
[ -f "$CKPT" ] || { echo "ERROR: ckpt missing"; exit 1; }
[ -f "$STATS" ] || { echo "ERROR: stats missing: $STATS"; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: manifest missing"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

NCHUNK=$((NGPU * PER_GPU))
ROUNDS="${ROUNDS:-3}"; STAGGER="${STAGGER:-10}"
# EGL_DEVICE: pin MuJoCo/robosuite EGL rendering of ALL workers to this CUDA-visible device index (compute stays
# on each worker's own GPU). 08-28 finding on gnho006: workers rendering on some GPUs SIGABRT in mjr_readPixels,
# only one GPU is reliable -> pin. robosuite asserts MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES, hence the pair.
EGL_DEVICE="${EGL_DEVICE:-}"
gpu_env() {  # args: gpu -> prints env assignments for the worker
  if [ -n "$EGL_DEVICE" ] && [ "$EGL_DEVICE" != "$1" ]; then echo "CUDA_VISIBLE_DEVICES=$1,$EGL_DEVICE MUJOCO_EGL_DEVICE_ID=$EGL_DEVICE";
  elif [ -n "$EGL_DEVICE" ]; then echo "CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$EGL_DEVICE";
  else echo "CUDA_VISIBLE_DEVICES=$1"; fi
}
export -f gpu_env

# EGL_DEVICE=auto: probe every visible GPU with exploration/action_img_patch/egl_probe_device.py (real LIBERO env,
# 3 x (create/reset/100 rendered steps/close)); pick the first device that passes, else leave rendering unpinned.
if [ "$EGL_DEVICE" = "auto" ]; then
  mkdir -p "$OUT/logs"; PASS=()
  for d in $(seq 0 $((NGPU - 1))); do
    ( CUDA_VISIBLE_DEVICES=$d MUJOCO_EGL_DEVICE_ID=$d timeout 900 python exploration/action_img_patch/egl_probe_device.py \
        > "$OUT/logs/egl_probe_dev$d.log" 2>&1; echo $? > "$OUT/logs/egl_probe_dev$d.rc" ) &
  done; wait
  for d in $(seq 0 $((NGPU - 1))); do
    rc=$(cat "$OUT/logs/egl_probe_dev$d.rc" 2>/dev/null || echo 99)
    echo "egl probe dev$d rc=$rc $(grep -c 'probe round' "$OUT/logs/egl_probe_dev$d.log" 2>/dev/null)/3 rounds"
    [ "$rc" = "0" ] && PASS+=("$d")
  done
  if [ ${#PASS[@]} -gt 0 ]; then EGL_DEVICE="${PASS[0]}"; else EGL_DEVICE=""; fi
  echo "=== EGL probe passed devices: ${PASS[*]:-none} -> EGL_DEVICE=${EGL_DEVICE:-unpinned}  $(date)"
fi
export EGL_DEVICE

remaining_tasks() {  # manifest rows without a results json
  local suite tid
  while IFS=, read -r suite tid; do
    [ -n "$suite" ] || continue
    [ -f "$OUT/$suite/gpu0_task${tid}_results.json" ] || echo "$suite,$tid"
  done < "$MANIFEST"
}

run_chunk() {  # args: round idx
  local round=$1 idx=$2 gpu=$(( $2 % NGPU )) chunk="$OUT/chunks/r${1}_chunk_$2.csv"
  [ -s "$chunk" ] || return 0
  local first_suite first_tid; IFS=, read -r first_suite first_tid < "$chunk"
  local log="$OUT/logs/r${round}_chunk_${idx}.log"
  sleep $(( idx * STAGGER ))
  env $(gpu_env $gpu) python -X faulthandler experiments/libero/eval_libero_single.py \
    --config-name sim_libero_omnigen2 task="$TASK" ckpt="$CKPT" gpu_id=0 seed=20260819 \
    EVALUATION.task_chunk_file="$chunk" EVALUATION.task_suite_name="$first_suite" EVALUATION.task_id="$first_tid" \
    EVALUATION.num_trials="$NUM_TRIALS" EVALUATION.output_dir="$OUT" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.action_horizon="$HORIZON" EVALUATION.replan_steps="$REPLAN" \
    model.flux2_src_path="$FLUX2" model.flux2_model_path="$FLUX_MODEL" model.ae_model_path="$AE_MODEL" \
    model.variant=klein-base-4b model.qwen3_model_spec="$QWEN_MODEL" model.load_text_encoder=true \
    model.pack_proprio_after_text=true model.proprio_dim=8 model.qwen_context_len=128 \
    > "$log" 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) round=$round chunk=$idx gpu=$gpu rows=$(wc -l < "$chunk") rc=$rc done=$(grep -c 'completed:' "$log")"
  [ $rc -eq 0 ] || echo "r${round}_chunk_${idx} rc=$rc" >> "$OUT/failed_chunks.txt"
}
export -f run_chunk
export OUT NGPU TASK CKPT NUM_TRIALS STATS HORIZON REPLAN FLUX2 FLUX_MODEL AE_MODEL QWEN_MODEL STAGGER EGL_DEVICE

TOTAL=$(grep -c . "$MANIFEST")
for round in $(seq 1 "$ROUNDS"); do
  REM="$OUT/chunks/r${round}_remaining.csv"; remaining_tasks > "$REM"
  NREM=$(grep -c . "$REM" || true)
  echo "=== round $round: remaining $NREM / $TOTAL  $(date)"
  [ "$NREM" -gt 0 ] || break
  rm -f "$OUT"/chunks/r${round}_chunk_*.csv
  awk -v n="$NCHUNK" -v d="$OUT/chunks" -v r="$round" 'NF { print > (d "/r" r "_chunk_" (NR-1) % n ".csv") }' "$REM"
  seq 0 $((NCHUNK - 1)) | xargs -P "$NCHUNK" -I{} bash -c "run_chunk $round {}"
  NOW=$(remaining_tasks | grep -c . || true)
  echo "=== round $round finished: remaining $NOW  $(date)"
  [ "$NOW" -lt "$NREM" ] || { echo "no progress in round $round; stop"; break; }
done

echo "=== all rounds done $(date); result json count: $(find "$OUT" -name '*_results.json' | wc -l) / $TOTAL"
python experiments/libero/summarize_results.py --output_dir="$OUT" 2>&1 | tail -40
echo "=== END $(date) OUT=$OUT"
