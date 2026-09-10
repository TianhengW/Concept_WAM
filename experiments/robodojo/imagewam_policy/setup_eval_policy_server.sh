#!/bin/bash
# Policy-server side for imagewam_policy. Same positional contract as demo_policy, but
# <policy_conda_env> may be either a conda env name OR a path to a Python venv
# (ImageWAM trains in /storage/yukaichengLab/mazijian/wth/ImageWAM/.venv, not conda).
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}, action_type=${action_type}, port=${policy_server_port}"

if [[ -f "${policy_conda_env}/bin/activate" ]]; then
    # python venv (e.g. ImageWAM/.venv)
    # shellcheck disable=SC1091
    source "${policy_conda_env}/bin/activate"
    echo "[SERVER] activated venv ${policy_conda_env}"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    echo "[SERVER] activated conda env ${policy_conda_env}"
fi

# XPolicyLab package + its client_server/utils modules, plus the RoboDojo root.
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
