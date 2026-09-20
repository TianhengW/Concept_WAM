#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -J ap_rc365_probe
#SBATCH -o slurm_logs/%x_%j.out
set -uo pipefail
cd /storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa
PY=/storage/yukaichengLab/lishiwen/xufanghui/xiaomi_robotics_1_RL/Xiaomi-Robotics-1/.conda-robocasa365/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
OUT=experiments/robocasa/eval/probe_logs; mkdir -p $OUT
for spec in "CloseBlenderLid 11 zero" "CloseBlenderLid 11 random" "CloseFridge 9 zero" "OpenDrawer 12 zero 750"; do
  set -- $spec; name="$1_$2_$3"
  echo "=== $spec ==="
  $PY experiments/robocasa/eval/probe_env.py $spec > $OUT/$name.log 2>&1; rc=$?
  last=$(grep -o "step=[0-9]*" $OUT/$name.log | tail -1)
  echo "rc=$rc last=$last $(grep -c 'Fatal Python error' $OUT/$name.log) fatal; $(grep 'COMPLETED' $OUT/$name.log)"
done
echo "=== gdb C backtrace for the first crashing spec (if gdb available) ==="
if command -v gdb >/dev/null; then
  gdb -batch -ex "run" -ex "bt 30" --args $PY experiments/robocasa/eval/probe_env.py CloseBlenderLid 11 zero 2>&1 | grep -v "^\[probe\] step=[0-9]*$" | tail -60
else echo "no gdb"; fi
