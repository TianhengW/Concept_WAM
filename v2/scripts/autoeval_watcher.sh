#!/usr/bin/env bash
# Auto-submit dedicated RoboTwin eval when a new training checkpoint appears (size-stable).
# Idempotent: per-step marker prevents duplicate submission.
WROOT=/storage/yukaichengLab/mazijian/wth/ImageWAM
EXP=$WROOT/v2/runs/robotwin_v2_armA/2026-08-01_02-08-56
WDIR=$EXP/checkpoints/weights
LOG=$WROOT/v2/slurm_logs/autoeval.log
TRAINJOB=80963
mkdir -p "$WDIR"
echo "$(date '+%F %T') watcher up, watching $WDIR (train job $TRAINJOB)" >> "$LOG"
idle=0
while true; do
  for f in "$WDIR"/step_*.pt; do
    [ -e "$f" ] || continue
    tag=$(basename "$f" .pt); step=${tag#step_}
    mk="$WDIR/.evalsub_$step"
    [ -f "$mk" ] && continue
    s1=$(stat -c %s "$f" 2>/dev/null || echo 0)
    sleep 25
    s2=$(stat -c %s "$f" 2>/dev/null || echo 1)
    if [ "$s1" = "$s2" ] && [ "$s1" != "0" ]; then
      cd "$WROOT"
      jid=$(sbatch --parsable --export=ALL,EVAL_TRAIN_STEP=$step,EXP_PATH=$EXP v2/scripts/sbatch_eval_dedicated.sbatch 2>>"$LOG")
      touch "$mk"
      echo "$(date '+%F %T') submitted eval job $jid for $tag" >> "$LOG"
    fi
  done
  if squeue -j "$TRAINJOB" -h 2>/dev/null | grep -q "$TRAINJOB"; then idle=0; else idle=$((idle+1)); fi
  if [ "$idle" -ge 20 ]; then echo "$(date '+%F %T') train job gone, watcher exit" >> "$LOG"; break; fi
  sleep 180
done
