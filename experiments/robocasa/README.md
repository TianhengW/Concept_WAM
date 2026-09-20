# action-as-patch × RoboCasa365

按官方 benchmark 标准打法:**pretrain(Human300)→ post-train(3 个 target split 各自微调)→
官方 sim 评测 50 target 任务**。参照 [RoboCasa365 论文](https://arxiv.org/abs/2603.04356) 与
[LeRobot robocasa 文档](https://huggingface.co/docs/lerobot/robocasa)。

## 前置依赖(repo 不含,需自备)

- **FLUX.2 源码**:backbone 里有 `from flux2.model import ...` / `from flux2.autoencoder import ...`
  (`src/imagewam/models/backbones/{mot,flux2_video_expert,flux2_three_stream_model}.py`),但本 repo
  **不包含** vendored 的 FLUX.2 源码。自备一份后用 `FLUX2_SRC` 指向它:`train_flux2_klein_imagewam.sh`
  会把 `${FLUX2_SRC}/src` 和 `${FLUX2_SRC}` 加入 PYTHONPATH,并以 `model.flux2_src_path=${FLUX2_SRC}`
  覆盖 config 里的默认值。也就是说 `${FLUX2_SRC}/src/flux2/` 或 `${FLUX2_SRC}/flux2/` 必须存在。
- **权重**:`AAP_FLUX2_KLEIN`(flux-2-klein-base-4b.safetensors)、`AAP_FLUX2_AE`(FLUX.2-dev/ae.safetensors)、
  `AAP_QWEN3`(Qwen3-4B)。把 `.env.example` 复制成 `.env.local` 填本机路径。
- **评测环境**:robocasa 1.0.1 + robosuite 1.5.2 + mujoco 3.3.1(独立 conda env,与训练 venv 分开),见 `eval/README.md`。

## 数据(已核验)

`/mnt/auwomo-data/auwomo-datasets/raw-data/manipulation/robocasa365`
- **LeRobot v2.1**(与 RoboTwin 线同构,dataloader 直接可用),PandaOmron,fps20
- 3 相机 256×256:`robot0_agentview_left / _right / _eye_in_hand`;state 16 维 / action 12 维
- pretrain 300 任务(65 atomic + 235 composite,~107 eps/任务)+ target 50 任务(500 eps/任务)
- 每任务路径 `<task>/<date>/lerobot/lerobot`(双层);MimicGen 在 `mg/`(Human300 不用)
- ⚠️ tar 已全量在位(350/350);**解压进行中**(66 并行进程),gen_dirs.sh 需在解压完成后重跑确认 300/18 roots 齐

## 口径设计(防 mismatch 的对齐依据)

- 相机布局:复用 RoboTwin 的 `concat_multi_camera: "robotwin"` compact 288×256
  (top=agentview_left,下=agentview_right|eye_in_hand)——与 RoboTwin 94.1 实验同一代码路径
- 归一化:z-score(processor 与 RoboTwin 线完全同配置);norm stats 用各任务 LeRobot meta 自带 stats
- 模型:`imagewam_flux2_klein_4b_actionpatch`(零参数 codec,action 12 维 → 12×10=120/128)
- DDP(ZERO_STAGE=0,集群 deepspeed nan bug);pretrain 队列 que-1b0ccceacaca;镜像复用 aap-c2r-train:v1

## 文件

- `configs/data/robocasa365.yaml`(数据)+ `robocasa365_*_dirs.yaml`(gen_dirs.sh 生成)
- `configs/task/robocasa_flux2_klein_4b_actionpatch_pretrain.yaml` / `..._posttrain.yaml`
- `gen_dirs.sh`(收集 LeRobot roots;composite seen/unseen 需先放 `seen_tasks.txt`/`unseen_tasks.txt` 官方清单)
- `smoke_robocasa.sh`(带显存采样)/ `env_robocasa.sh` / `vcjob_robocasa_pretrain_4node.yaml`
- `eval/`:评测骨架(见下)

## 跑法

```bash
bash gen_dirs.sh                       # 解压完成后生成/刷新 dirs yaml
SMOKE_BATCH=12 bash smoke_robocasa.sh  # 冒烟+测显存(目标>=80%=115G)
kubectl apply -f vcjob_robocasa_pretrain_4node.yaml

# post-train(pretrain 完成后,每 split 一次)
# 见 configs/task/robocasa_flux2_klein_4b_actionpatch_posttrain.yaml 头部注释
```

### 实际产出模型的那次 pretrain(H800 Slurm,2 节点 × 8 卡)

```bash
BATCH=16 sbatch experiments/robocasa/sbatch_robocasa_pretrain_2node.sh
```

- `BATCH` 是**必填**的每卡 batch(脚本里 `${BATCH:?}`),不传会直接退出;task config 里的
  `batch_size: 24` 是 diguayun 140G 卡的默认值,H800 上用 16 → global 16×16 = **256**。
- 脚本内固定:`ZERO_STAGE=1`(ZeRO-1)、rendezvous 走 192.168.x 网卡(`eno1` / `enp86s0f0np0`)、
  `WANDB_MODE=offline`、`HF_HUB_OFFLINE=1`。
- config 侧:lr 1e-4 cosine、`num_epochs: 5` = **559,660 步**(~1.06 s/step,16 卡约 6.9 天)、
  每 5000 步存 weights + 最新一份 optimizer/dataloader state。
- 两个指定节点被占时,用 `sbatch_robocasa_pretrain_2node_anynode.sh`(仅去掉 `-w` 绑定)。
- 续训:同样的脚本加 hydra 覆盖,指向 **state 目录**(不是 weights 的 .pt):

  ```bash
  BATCH=16 sbatch experiments/robocasa/sbatch_robocasa_pretrain_2node_anynode.sh \
    resume=<run_dir>/checkpoints/state/step_485000
  ```

## 评测(eval/)

评测流水线已实现,细节(环境、渲染后端、崩溃恢复、任务拆分)见 **`eval/README.md`**。
一句话结构:每张卡一对 `ap_policy_server.py`(AP venv)+ `eval_robocasa_ap.py`(robocasa conda env),
任务集与 horizon 取自 `robocasa.utils.dataset_registry`,结果按任务写 `stats.json`,
`merge_summaries.py` 合并成 `summary.json`。

```bash
RUN=runs/robocasa_flux2_klein_4b_actionpatch_pretrain/<timestamp>
CKPT=$RUN/checkpoints/weights/step_559660.pt RUN_DIR=$RUN \
TASK_SET=atomic_seen NUM_TRIALS=50 SPLIT=pretrain \
  sbatch --gres=gpu:8 experiments/robocasa/eval/sbatch_eval_robocasa.sh
```

口径与官方 harness(`robocasa-benchmark/openpi` 的 `examples/robocasa/main.py`)一致:
`split=pretrain`、horizon 取 `get_task_horizon`、16 步 action chunk 开环执行后重推理、
每个仿真步检查一次 `info["success"]`、每任务 50 次(composite 视预算可降;官方复测用 30)。
`eval/eval_robocasa_single.py` 是早期骨架,已被 `eval_robocasa_ap.py` 取代。
