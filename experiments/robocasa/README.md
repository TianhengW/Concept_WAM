# action-as-patch × RoboCasa365

按官方 benchmark 标准打法:**pretrain(Human300)→ post-train(3 个 target split 各自微调)→
官方 sim 评测 50 target 任务**。参照 [RoboCasa365 论文](https://arxiv.org/abs/2603.04356) 与
[LeRobot robocasa 文档](https://huggingface.co/docs/lerobot/robocasa)。

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

## 评测(eval/)

官方评测在 robocasa sim(robosuite 系)中 rollout 50 target 任务。接入步骤(TODO):
1. 安装官方 [robocasa](https://github.com/robocasa/robocasa) + robosuite 环境(参考其 setup 文档)
2. policy 适配:仿 `experiments/libero/eval_libero_single.py` 的结构写 robocasa 版
   (obs 3 相机 → compact 拼接 → imagewam policy → 12 维 action 执行)
3. 按 leaderboard 口径:三个 split 分别评,报 SR
骨架文件 `eval/eval_robocasa_single.py` 已给出接口注释。
