#!/bin/bash
# XPolicyLab naming stub. ImageWAM trains through Slurm, not through this script:
#   sbatch exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh
# Args (contract): <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>
set -euo pipefail
echo "[train.sh] imagewam_policy trains via ImageWAM/exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh"
echo "[train.sh] expected checkpoint layout for eval: checkpoints/<ckpt_name>/<step>.pt (+ dataset_stats.json)  args=$*"
