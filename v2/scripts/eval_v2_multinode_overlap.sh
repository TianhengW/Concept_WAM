#!/usr/bin/env bash
# Full RoboTwin eval of step_005000, spread across the 4 training nodes' spare
# VRAM via srun --overlap (one 8-GPU manager per node, ~1/4 of the tasks each).
# 50 tasks x [clean,random] x 10 episodes. No arrays (portable / robust).
set -o pipefail

JOBID="${JOBID:-79556}"
EVAL=/storage/yukaichengLab/mazijian/wth/ImageWAM/v2/scripts/eval_v2_robotwin.sh
LOGDIR=/storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs
mkdir -p "${LOGDIR}"

run_node() {
  node="$1"
  tasks="$2"
  n=$(printf '%s' "$tasks" | tr ',' '\n' | grep -c .)
  echo "[launch] node=${node} n_tasks=${n}"
  srun --overlap --jobid="${JOBID}" -w "${node}" --ntasks=1 bash -c \
    "NUM_GPUS=8 MAX_TASKS_PER_GPU=2 EVAL_NUM_EPISODES=10 PHASES=[clean,random] TASK_NAME='${tasks}' bash ${EVAL}" \
    > "${LOGDIR}/eval_full_${node}.log" 2>&1 &
  sleep 5
}

run_node gnho006 "adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,lift_pot,move_can_pot,move_playingcard_away"
run_node gnho011 "move_stapler_pad,hanging_mug,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_can_basket,place_cans_plasticbox,place_container_plate"
run_node gnho036 "place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_mouse_pad,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler"
run_node gnho041 "put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,stamp_seal,turn_switch"

echo "[launch] all 4 node-managers started; waiting..."
wait
echo "ALL_DONE"
