#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho032
#SBATCH -J v2eval
#SBATCH -o /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/slurm_logs/%x_%j.out
set -euo pipefail
# Dedicated-node eval (no training overlap): 2 tasks/GPU. EVAL_TRAIN_STEP via --export.
: "${EVAL_TRAIN_STEP:?set EVAL_TRAIN_STEP}"
TASKS47="adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic,lift_pot,move_can_pot,move_playingcard_away,hanging_mug,open_laptop,open_microwave,pick_diverse_bottles,pick_dual_bottles,place_a2b_left,place_a2b_right,place_bread_basket,place_bread_skillet,place_can_basket,place_cans_plasticbox,place_container_plate,place_dual_shoes,place_empty_cup,place_fan,place_burger_fries,place_object_basket,place_object_scale,place_object_stand,place_phone_stand,move_pillbottle_pad,place_shoe,press_stapler,put_bottles_dustbin,put_object_cabinet,rotate_qrcode,scan_object,shake_bottle,shake_bottle_horizontally,stack_blocks_three,stack_blocks_two,stack_bowls_three,stack_bowls_two,turn_switch"
NUM_GPUS=8 MAX_TASKS_PER_GPU="${MAX_TASKS_PER_GPU:-2}" EVAL_NUM_EPISODES=10 PHASES="[clean,random]" \
  TASK_NAME="${TASKS47}" \
  bash /storage/yukaichengLab/mazijian/wth/ImageWAM/v2/scripts/eval_v2_robotwin.sh
