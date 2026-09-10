#!/usr/bin/env bash
# Self-healing wrapper: resubmit robodojo_eval_smoke.sbatch, adding any node that
# reports DRIVER_MISMATCH (NVIDIA 6xx driver, Kit/Vulkan crash) to the exclude list.
set -uo pipefail
R=/storage/yukaichengLab/mazijian/wth/ImageWAM
S=/storage/yukaichengLab/share/datasets/.robodojo_state
EXCL="nv-h100-029,nv-h100-007,gnho006,gnho041,gnho035"
for attempt in $(seq 1 12); do
  echo "[retry] attempt ${attempt} exclude=${EXCL} $(date)"
  jid=$(/soft/slurm/bin/sbatch --parsable --export=ALL --time=01:00:00 --exclude="${EXCL}" "$R/scripts/robodojo_eval_smoke.sbatch")
  echo "[retry] submitted ${jid}"
  while /soft/slurm/bin/squeue -j "${jid}" -h -o %T | grep -qE "PENDING|RUNNING|CONFIGURING|COMPLETING"; do sleep 60; done
  sleep 5
  log="$S/eval_smoke_${jid}.out"
  if grep -q "DRIVER_MISMATCH" "${log}" 2>/dev/null; then
    node=$(grep -m1 "DRIVER_MISMATCH" "${log}" | grep -oE "host=[^ ]+" | cut -d= -f2)
    echo "[retry] ${jid} hit 6xx driver on ${node}; excluding"
    EXCL="${EXCL},${node}"
    continue
  fi
  st=$(/soft/slurm/bin/sacct -j "${jid}" -X -n -o State | head -1 | tr -d " ")
  echo "[retry] ${jid} finished state=${st}; MDL_ERRORS=$(grep -cE "MDLC|could not find module" "${log}" 2>/dev/null)"
  grep -E "Success nums|finished status|giving up" "${log}" | head -3
  echo "RETRY_DONE jid=${jid} state=${st}"
  exit 0
done
echo "RETRY_EXHAUSTED exclude=${EXCL}"
exit 1
