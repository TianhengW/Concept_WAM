# action_img_patch — 动作当图像 patch(单流统一)

**Idea**:不要 action expert。action chunk 通过**固定的零参数编解码**映射到与 FLUX.2
packed latent 相同的 128 维 token 空间(14 维动作每维平铺 9 次占 126 维 + 补 2 零,
解码取平铺均值),拼进 noisy target 图像序列,由**原封不动的 FLUX.2 主干**当作普通
图像 patch 一起去噪。一条流、一个 sigma、同一个 flow-matching loss(在图像 token 和
action token 上分别统计,权重 λv=0.5 / λa=1.0,与 baseline 相同)。

```
joint sequence: [txt(+proprio) | ref(clean) | target(noisy) | action(noisy)]
                                              \______ 同一个 sigma ______/
```

掩码 = 原版 ImageWAM flux2 掩码取 `action_len=0`(action token 并入 noisy target 块),
即只剩 clean/noisy 区分:clean 前缀不看 noisy,noisy 看全部。

**新增可训练参数:0**。可训练的就是 FLUX.2 transformer 本身(+共享的 proprio encoder,
与 baseline 一致)。ckpt payload 仍是 `{"mot", "proprio_encoder", ...}`,trainer 保存/
恢复无需改动。

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | `ImageWAMActionPatch(ImageWAM)`:固定 codec、video-only MoT 前向、单 sigma 训练 loss、联合去噪 `infer_action_flux2` |
| `factory.py` | hydra `_target_` 工厂 |
| `smoke_test.py` | 1 卡冒烟:构建/前向/反传/codec 往返/推理形状 |
| `sbatch_actionpatch_2node.sh` | 本实验 2 节点训练 |
| `sbatch_baseline_2node.sh` | baseline(原始 ImageWAM MoT,concept_k=0)2 节点训练 |
| `../../configs/model/imagewam_flux2_klein_4b_actionpatch.yaml` | 模型配置(指向本目录 factory) |
| `../../configs/task/robotwin_flux2_klein_4b_actionpatch_sub20.yaml` | 本实验 task |
| `../../configs/task/robotwin_flux2_klein_4b_base_sub20.yaml` | baseline task |

## 公平对比设定(两边完全一致)

- 数据:fastwam_robotwin/robotwin2.0 + nonidle 过滤 + release norm stats,
  **`sample_index_stride=20`(1/20 抽样,现成开关:`__len__=ceil(N/20)`,`idx*20` 取样)**
- batch_size=4 × grad_accum=2 × 16 卡(2 节点)= **global 128**
- lr 1e-4 cosine / wd 1e-2 / bf16 / 10 epochs / save_every 2500
- text 条件:在线 Qwen3-4B 编码(`load_text_encoder=true`,ctx 128),无 reasoner
- 唯一差异 = 模型:baseline 是 MoT(FLUX.2 video expert + 独立 action DiT,
  concept_k=0 即原始行为);本实验是单流 action-as-patch

## 提交

```bash
cd /storage/yukaichengLab/mazijian/wth/ImageWAM
sbatch exploration/action_img_patch/sbatch_baseline_2node.sh
sbatch exploration/action_img_patch/sbatch_actionpatch_2node.sh
```

## 已知取舍 / 注意

- **推理**是联合去噪(future image + action 一起,20 步),不能像 MoT 那样 KV-cache
  只解 action——action 天然 grounded 在预测未来上,这是本 idea 的特性也是代价。
  评测接入时 `infer_action` 返回 `{"action": [H,14] cpu float32}`,与 MoT 版接口一致。
- codec 是固定的(不可学习),不存在塌缩问题,loss 保持纯 flow matching;
  若要试可学习 adapter,后续在 `action_patch_scale`/codec 处扩展。
- `action_dim_is_pad`(逐维 pad)在平铺 token 空间不适用(RoboTwin 也不用);
  逐步 pad(`action_is_pad`)照常生效。
