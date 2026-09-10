#!/usr/bin/env bash
# Standard LIBERO evaluation (4 suites x 10 tasks) for action-as-patch ckpts on H800.
# Runtime = xz's python env (torch 2.7.1 / mujoco 3.3.2 / robosuite 1.4.0, transformers 4.51 overlay)
# + upstream LIBERO checkout third_party/LIBERO (torch.load weights_only patch applied).
# Workers: one eval_libero_single.py process per (suite, task_id), PER_GPU concurrent per GPU (no tmux).
# Env knobs:
#   CKPT   (required) .../runs/<task>/<date>/checkpoints/weights/step_N.pt
#   STATS  dataset_stats.json used at training (default: <run_dir>/dataset_stats.json)
#   TASK   hydra task config (default libero_flux2_klein_4b_actionpatch)
#   SUITES space-separated (default "libero_spatial libero_object libero_goal libero_10")
#   TASK_IDS space-separated (default 0..9);  NUM_TRIALS (default 50);  NGPU (default 8);  PER_GPU (default 2)
#   REPLAN (default 12); HORIZON (default 16); OUT_TAG (default eval_<jobid>); ROUNDS (3); STAGGER (10 s between worker starts)
# Units that fail (e.g. the start-up-burst SIGABRT seen on 08-28) are re-run in later rounds; units with a results json are skipped.
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -x gnho006,gnho020,gnho041
#SBATCH -J eval_libero_std
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail
REPO=/storage/yukaichengLab/mazijian/wth/ImageWAM
cd "$REPO"; mkdir -p slurm_logs
ENV=/storage/yukaichengLab/mazijian/xz/runtime/libero_plus_h800_native_20260820_r1
SUP=/storage/yukaichengLab/mazijian/xz/runtime/libero_plus_h800_native_support_20260821_r3
FLUX2=/storage/yukaichengLab/mazijian/wth/flux2
LIBERO_DIR="$REPO/third_party/LIBERO"
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
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASK_IDS="${TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NGPU="${NGPU:-8}"; PER_GPU="${PER_GPU:-2}"
REPLAN="${REPLAN:-12}"; HORIZON="${HORIZON:-16}"
CKPT_TAG="$(basename "$(dirname "$RUN_DIR")")_$(basename "$RUN_DIR")"   # <task>_<date> (matches eval_libero_single tagging)
OUT_TAG="${OUT_TAG:-eval_${SLURM_JOB_ID}_$(basename "$CKPT" .pt)}"
OUT="$REPO/evaluate_results/libero_std/$CKPT_TAG/$OUT_TAG"
mkdir -p "$OUT/logs"

export PATH="$ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SUP/python-overlay:$SUP/transformers-overlay/transformers451-r1:$LIBERO_DIR:$REPO:$REPO/src:$FLUX2/src:$FLUX2"
export LIBERO_CONFIG_PATH="$REPO/experiments/libero/libero_std_config"
export LD_LIBRARY_PATH="/storage/soft/cuda/cuda-12.5/lib64:$ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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

echo "=== START $(date) node=$SLURMD_NODENAME ckpt=$CKPT stats=$STATS task=$TASK trials=$NUM_TRIALS out=$OUT"
[ -f "$CKPT" ] || { echo "ERROR: ckpt missing"; exit 1; }
[ -f "$STATS" ] || { echo "ERROR: stats missing: $STATS"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

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
run_unit() {  # args: round idx suite task_id
  local round=$1 idx=$2 suite=$3 tid=$4 gpu=$(( $2 % NGPU ))
  [ -f "$OUT/$suite/gpu0_task${tid}_results.json" ] && return 0
  local log="$OUT/logs/r${round}_${suite}_task${tid}.log"
  sleep $(( (idx % (NGPU * PER_GPU)) * STAGGER ))
  env $(gpu_env $gpu) python -X faulthandler experiments/libero/eval_libero_single.py \
    --config-name sim_libero_omnigen2 task="$TASK" ckpt="$CKPT" gpu_id=0 \
    EVALUATION.task_suite_name="$suite" EVALUATION.task_id="$tid" \
    EVALUATION.num_trials="$NUM_TRIALS" EVALUATION.output_dir="$OUT" \
    EVALUATION.dataset_stats_path="$STATS" \
    EVALUATION.action_horizon="$HORIZON" EVALUATION.replan_steps="$REPLAN" \
    model.flux2_src_path="$FLUX2" model.flux2_model_path="$FLUX_MODEL" model.ae_model_path="$AE_MODEL" \
    model.variant=klein-base-4b model.qwen3_model_spec="$QWEN_MODEL" model.load_text_encoder=true \
    model.pack_proprio_after_text=true model.proprio_dim=8 model.qwen_context_len=128 \
    > "$log" 2>&1
  local rc=$?
  echo "$(date +%H:%M:%S) round=$round unit=$suite/$tid gpu=$gpu rc=$rc"
  [ $rc -eq 0 ] || echo "r${round} $suite,$tid rc=$rc" >> "$OUT/failed_units.txt"
}
export -f run_unit
export OUT NGPU PER_GPU TASK CKPT NUM_TRIALS STATS HORIZON REPLAN FLUX2 FLUX_MODEL AE_MODEL QWEN_MODEL STAGGER EGL_DEVICE

UNITS=(); for s in $SUITES; do for t in $TASK_IDS; do UNITS+=("$s $t"); done; done
for round in $(seq 1 "$ROUNDS"); do
  TODO=(); for u in "${UNITS[@]}"; do set -- $u; [ -f "$OUT/$1/gpu0_task${2}_results.json" ] || TODO+=("$u"); done
  echo "=== round $round: ${#TODO[@]} / ${#UNITS[@]} units remaining  $(date)"
  [ "${#TODO[@]}" -gt 0 ] || break
  i=0; for u in "${TODO[@]}"; do echo "$round $i $u"; i=$((i+1)); done | xargs -P $((NGPU * PER_GPU)) -L 1 bash -c 'run_unit $0 $1 $2 $3'
  LEFT=0; for u in "${UNITS[@]}"; do set -- $u; [ -f "$OUT/$1/gpu0_task${2}_results.json" ] || LEFT=$((LEFT+1)); done
  echo "=== round $round finished: $LEFT remaining  $(date)"
  [ "$LEFT" -lt "${#TODO[@]}" ] || { echo "no progress in round $round; stop"; break; }
done

echo "=== all rounds done $(date); result json count: $(find "$OUT" -name '*_results.json' | wc -l)"
python experiments/libero/summarize_results.py --output_dir="$OUT" 2>&1 | tail -40
echo "=== END $(date) OUT=$OUT"
