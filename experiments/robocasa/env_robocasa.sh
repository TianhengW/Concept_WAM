#!/usr/bin/env bash
# Diguayun environment for the action-as-patch RoboCasa365 experiments.

export AAP_ROBOCASA_ROOT=/mnt/auwomo-data/auwomo-datasets/raw-data/manipulation/robocasa365
export AAP_FLUX2_KLEIN=/mnt/auwomo-data/auwomo-model/ImageWAM/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors
export AAP_FLUX2_AE=/mnt/auwomo-data/auwomo-model/ImageWAM/FLUX.2-dev/ae.safetensors
export AAP_QWEN3=/mnt/auwomo-data/auwomo-model/Qwen3-4B

export FLUX2_SRC=/mnt/auwomo-data/wangtianheng/codes/action-as-patch

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# DDP, not deepspeed (deepspeed engine NaNs on this cluster; see C2R notes)
export ZERO_STAGE=0
