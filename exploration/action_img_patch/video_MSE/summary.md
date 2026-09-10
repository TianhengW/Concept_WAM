# 批量 video-MSE 对比:官方 MoT baseline vs patch 90k

- 样本: **75 个** = dataset 均匀切 15 块 × 每块 5 个相邻样本(近似同场景多 episode/帧窗口)
- val dataset 无 task_id,仅自然语言 instruction;块 label 取指令关键词(`bNN_...`)
- 指标: 未来帧 MSE(pred, GT),图像空间 [-1,1],同 `infer_video_flux2` / 20 步 / seed=0
- baseline: 官方 HF `ImageWAM-FLUX.2-4B-RoboTwin/model.pt`(MoT,step 69900);patch: step_090000

| 总体(N=75) | MSE |
|---|---:|
| **patch 90k** | **0.0154** |
| **官方 MoT baseline** | **0.0113** |
| naive(copy-current) | 0.0996 |
| patch vs baseline | -36.0% |
| 逐样本 patch ≤ baseline | 39/75 |

## 逐块均值

| block | n | baseline MSE | patch MSE | patch 优势 |
|---|---:|---:|---:|---:|
| b00_Grab the white fan using | 5 | 0.0078 | 0.0086 | -11% |
| b01_Position the plastic sta | 5 | 0.0094 | 0.0110 | -18% |
| b02_Place the ceramic bowl d | 5 | 0.0013 | 0.0013 | -1% |
| b03_Lift the teal-bottom bot | 5 | 0.0042 | 0.0062 | -49% |
| b04_With the right arm, lift | 5 | 0.0019 | 0.0020 | -7% |
| b05_Take the sound microphon | 5 | 0.0058 | 0.0048 | +17% |
| b06_Use the arm to set the c | 5 | 0.0016 | 0.0008 | +48% |
| b07_With the right arm, the  | 5 | 0.0030 | 0.0031 | -5% |
| b08_Slide open the drawer of | 5 | 0.0417 | 0.0407 | +2% |
| b09_Move large block, medium | 5 | 0.0077 | 0.0077 | -0% |
| b10_First move the cylindric | 5 | 0.0787 | 0.1368 | -74% |
| b11_After grabbing the red p | 5 | 0.0017 | 0.0016 | +5% |
| b12_Move the container with  | 5 | 0.0012 | 0.0023 | -100% |
| b13_Pick up the small soft h | 5 | 0.0010 | 0.0007 | +28% |
| b14_Lift the gray tabletrash | 5 | 0.0025 | 0.0025 | +1% |

- 逐块 patch ≤ baseline: **6/15**