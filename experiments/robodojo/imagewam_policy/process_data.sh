#!/bin/bash
# XPolicyLab naming stub. Data conversion lives in ImageWAM:
#   sbatch scripts/data/sbatch_robodojo_convert.sh --tasks complete --out-root /storage/yukaichengLab/share/datasets/RoboDojo_lerobot
# Args (contract): <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
set -euo pipefail
echo "[process_data.sh] imagewam_policy converts RoboDojo hdf5 -> LeRobot v2.1 via ImageWAM/scripts/data/robodojo_raw_to_lerobot.py  args=$*"
