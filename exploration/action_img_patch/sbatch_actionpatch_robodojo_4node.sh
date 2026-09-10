#!/usr/bin/env bash
#SBATCH -p yukaichenglab
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH -x nv-h100-029,nv-h100-007,gnho006,gnho020,gnho041,gnho034,gnho036,gnho033
#SBATCH -J ap_robodojo_4n
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH --requeue
# action-as-patch on RoboDojo (35 tasks x 100 ep), 4 nodes x 8 GPUs, global batch 256.
# Same launch recipe as sbatch_actionpatch_full_4node.sh; only TASK / job name / port differ.
# Resume: sbatch ... sbatch_actionpatch_robodojo_4node.sh resume=runs/<wandb name>/<ts>/checkpoints/state/step_NNNNNN
set -euo pipefail
REPO_ROOT="/storage/yukaichengLab/mazijian/wth/ImageWAM"
cd "${REPO_ROOT}"
mkdir -p slurm_logs
source .venv/bin/activate
export PATH="/soft/slurm/bin:${PATH}"

export NCCL_SOCKET_IFNAME=eno1
export GLOO_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_4,mlx5_6,mlx5_8   # ACTIVE 400G NDR rails only (skip 200G mlx5_2 + DOWN rails)
export NCCL_DEBUG=WARN

# --- 通信调优 (2026-09-05) ---
# 诊断: 4 节点 32 卡下 IB 实测仅 6.37 GB/s = 4x400Gb/s 链路的 3.2%；
#   每步梯度流量 24.5GB / 6.37GB/s = 3.85s ~= 实测步间隔 3.8s，
#   dmon 显示 70% 时间 sm=100% 但显存带宽 0%、功率 124W（NCCL spin-wait）。
# 症状是"梯度被拆成大量小消息，跑不满高速链路"，以下针对性放大消息粒度与并发。
export IMAGEWAM_DDP_BUCKET_MB=512      # DDP 默认 25MB -> 512MB，减少 all-reduce 次数
export NCCL_IB_QPS_PER_CONNECTION=4    # 每连接多 QP，提升单 rail 并发
export NCCL_IB_SPLIT_DATA_ON_QPS=1     # 把单次传输拆到多个 QP 上并行
export NCCL_BUFFSIZE=8388608           # 8MB 通道缓冲 (默认 4MB)
export NCCL_MIN_NCHANNELS=8            # 提高通道数以喂满 4 条 rail
export NCCL_NET_GDR_LEVEL=SYS          # 放开 GPUDirect RDMA (peermem 已加载)
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# FlightRecorder: 记录每个 rank 最近的 NCCL work，超时时 dump 出卡住 rank 的真实调用栈。
# 98835/99787 两次挂死都因为它没开，只能靠"哪个 rank 没报超时"反推元凶(两次都是 gnho033)。
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=/storage/yukaichengLab/mazijian/wth/ImageWAM/slurm_logs/nccl_trace/${SLURM_JOB_ID:-0}_rank_
# Enlarge the collective/watchdog timeout so a transient multi-node straggle (a rank
# lagging on one ALLREDUCE) does not abort the whole 4-node job. 95301 died on the
# 10min NCCL default; give 45min. IMAGEWAM_PG_TIMEOUT_MIN feeds Accelerate InitProcessGroupKwargs.
export IMAGEWAM_PG_TIMEOUT_MIN=45
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=2700
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ZERO_STAGE=1
export FLUX2_SRC="/storage/yukaichengLab/mazijian/wth/flux2"

# eno1 IPv4 of the first node for rendezvous (compute hostnames are IPv6 link-local only).
FIRST_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "${FIRST_NODE}" bash -c "ip -o -4 addr show eno1 | awk '{print \$4}' | cut -d/ -f1")
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29541}"

srun --export=ALL --kill-on-bad-exit=1 bash -c '
  export NNODES="${SLURM_NNODES}"
  export NODE_RANK="${SLURM_NODEID}"
  export PYTHONPATH="'"${REPO_ROOT}"':${PYTHONPATH:-}"
  export IMAGEWAM_PG_TIMEOUT_MIN="${IMAGEWAM_PG_TIMEOUT_MIN:-45}"
  export TASK=robodojo_flux2_klein_4b_actionpatch_full
  GPU_PER_NODE=8 bash scripts/flux2/train_flux2_klein_imagewam.sh 8 \
    wandb.mode=offline "$@"
' _ "$@"
