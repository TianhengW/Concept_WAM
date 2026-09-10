# imagewam_policy (ImageWAM action-as-patch, RoboDojo / arx_x5, joint control)

XPolicyLab adapter for the ImageWAM single-stream "action-as-patch" policy trained with
`configs/task/robodojo_flux2_klein_4b_actionpatch_full.yaml`. Lives in
`ImageWAM/experiments/robodojo/imagewam_policy/` and is symlinked to
`third_party/RoboDojo/XPolicyLab/policy/imagewam_policy` (RoboDojo only needs `eval.sh` + `deploy.yml`).

Inference reuses `experiments/robotwin/imagewam_policy/deploy_policy.py` (`WorldActionRobotWinPolicy`):
three cameras tiled into the compact 288x256 canvas, 14-d joint state z-scored with the run's
`dataset_stats.json`, 16-step joint action chunk denoised jointly with the future frame.

## Data / training (ImageWAM side, not part of XPolicyLab)
```bash
sbatch scripts/data/sbatch_robodojo_convert.sh --tasks complete --out-root /storage/yukaichengLab/share/datasets/RoboDojo_lerobot
.venv/bin/python scripts/data/compute_robotwin_nonidle_ranges.py <root> --output <root>/nonidle_ranges.json
.venv/bin/python scripts/data/compute_robodojo_dataset_stats.py --dataset-dir <root>
sbatch exploration/action_img_patch/sbatch_robodojo_train_smoke.sh          # 30 steps, 1 node
sbatch exploration/action_img_patch/sbatch_actionpatch_robodojo_4node.sh    # full run
```

## Evaluation
Policy env = the ImageWAM venv (a path is accepted wherever XPolicyLab expects a conda env name).
Checkpoint: either export `IMAGEWAM_CKPT=/…/checkpoints/weights/step_100000.pt` (stats found in the
run dir automatically) or symlink it as `checkpoints/<ckpt_name>/step.pt` next to this README.

```bash
cd third_party/RoboDojo
export IMAGEWAM_CKPT=/storage/yukaichengLab/mazijian/wth/ImageWAM/runs/robodojo_flux2_klein_4b_actionpatch_full/<ts>/checkpoints/weights/step_100000.pt
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/imagewam_policy --task stack_bowls \
     --ckpt ap_rd_100k --policy-env /storage/yukaichengLab/mazijian/wth/ImageWAM/.venv \
     --action-type joint --eval-num 1                     # smoke
bash scripts/robodojo.sh benchmark --policy-dir XPolicyLab/policy/imagewam_policy \
     --ckpt ap_rd_100k --policy-env /storage/yukaichengLab/mazijian/wth/ImageWAM/.venv \
     --action-type joint --eval-num native --gpu-ids 0,1,2,3,4,5,6,7   # official protocol
bash scripts/robodojo.sh summarize
```
`--action-type joint` is mandatory (robodojo.sh defaults to `ee`). Offline wiring check without
Isaac Sim: `EVAL_ENV_TYPE=debug bash scripts/robodojo.sh eval ...`.
