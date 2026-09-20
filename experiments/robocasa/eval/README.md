# RoboCasa365 evaluation for Action-as-Patch (H800 cluster)

Client–server split, one GPU per pair:

* **`ap_policy_server.py`** — runs in the AP training venv (`mazijian/wth/ImageWAM/.venv`), loads
  `<run_dir>/config.yaml` + a `checkpoints/weights/step_XXXXXX.pt`, serves action chunks over a local
  socket. Preprocessing mirrors training exactly (per-cam Resize → "robotwin" compact 288×256 concat →
  [-1,1]; proprio z-score via the run's `dataset_stats.json`; `DEFAULT_PROMPT`).
* **`eval_robocasa_ap.py`** — runs in the robocasa conda env
  (`lishiwen/.../Xiaomi-Robotics-1/.conda-robocasa365`: robocasa 1.0.1 / robosuite 1.5.2 / mujoco 3.3.1).
  Official task sets from `robocasa.utils.dataset_registry.TASK_SET_REGISTRY` (`target50`, `atomic_seen`,
  `composite_seen`, `composite_unseen`, `pretrain300`, …), horizons from `get_task_horizon`, seeded resets.
  The 12-d action is mapped to the env's dict action **by name via `modality.json`** (the LeRobot flat order
  `base_motion, control_mode, eef_pos, eef_rot, gripper` differs from `robocasa.utils.env_utils.convert_action`).
* **`run_eval_pair.sh`** — starts one server + one client on a GPU, restarts the client on native crashes.
* **`sbatch_eval_robocasa.sh`** — one node, two layouts (see Robustness for why):
  * `RENDER_BACKEND=egl` (default): `N - N_RENDER` policy pairs + `N_RENDER` GPUs used **only** for rendering,
    one client per render GPU (`N_RENDER` defaults to `N/2`: 8 GPUs → 4 pairs). ~1 min per 900-step episode.
  * `RENDER_BACKEND=osmesa`: clients render on CPU (OSMesa 24.0.7 from `mazijian/envs/osmesa_24.0.7`), every
    GPU hosts a policy pair (8 GPUs → 8 pairs). ~0.15–0.25 s/step → 2–4 min per episode; needs
    `--cpus-per-task` ≈ 8 per pair. Node throughput is similar for both; osmesa has no driver dependency.
  Tasks are round-robin over workers; results are merged into `summary.json` by `merge_summaries.py`.
  The scripts are snapshotted into `<run>/code/` at job start (bash reads scripts incrementally — never edit
  the repo copy of `run_eval_pair.sh` while a job is running).

## Run

```bash
cd /storage/yukaichengLab/mazijian/wth/action-as-patch-robocasa
RUN=runs/robocasa_flux2_klein_4b_actionpatch_pretrain/2026-09-12_19-58-48

# full official protocol: 50 target tasks x 50 trials, 8 GPUs (4 policy pairs + 4 render GPUs), ~10 h
CKPT=$RUN/checkpoints/weights/step_100000.pt RUN_DIR=$RUN TASK_SET=target50 NUM_TRIALS=50 \
  sbatch --gres=gpu:8 --cpus-per-task=32 experiments/robocasa/eval/sbatch_eval_robocasa.sh

# same on CPU rendering: 8 policy pairs, ~10 h, no render GPUs
CKPT=... RUN_DIR=$RUN TASK_SET=target50 NUM_TRIALS=50 RENDER_BACKEND=osmesa \
  sbatch --gres=gpu:8 --cpus-per-task=64 experiments/robocasa/eval/sbatch_eval_robocasa.sh

# one split, fewer trials, failure videos
CKPT=... RUN_DIR=$RUN TASK_SET=atomic_seen NUM_TRIALS=10 EXTRA="--save-failure-videos" \
  sbatch --gres=gpu:4 experiments/robocasa/eval/sbatch_eval_robocasa.sh

# smoke: two tasks, two trials, 2 GPUs (1 pair + 1 render GPU)
CKPT=... RUN_DIR=$RUN TASKS="OpenDrawer CloseFridge" NUM_TRIALS=2 \
  sbatch --gres=gpu:2 experiments/robocasa/eval/sbatch_eval_robocasa.sh
```
Per-task splits for leaderboard-style reporting: `TASK_SET=atomic_seen | composite_seen | composite_unseen`
(18 / 16 / 16 tasks = `target50`).

Env knobs: `TASK_SET`, `TASKS` (space-separated subset), `NUM_TRIALS`, `SPLIT` (`pretrain` = training
object instances, `target` = held-out), `REPLAN` (default 16 = full chunk), `EXTRA` (client flags),
`RUN_ID`, `OUT_ROOT`, `NUM_INFERENCE_STEPS` (server, default 10).
Results: `eval_results/robocasa365/<RUN_ID>/{summary.json, <task>/stats.json, <task>/*.mp4, logs/}`.

## Robustness

**Root cause of the native aborts (found with gdb):** the NVIDIA EGL driver (`libnvidia-eglcore.so.610.43.02`
on the gnho nodes) calls `abort()` inside `mjr_readPixels` when the sim client renders on a GPU that also
hosts a CUDA process (the policy server). Policy-free probes on an otherwise idle GPU never crash; with the
server on the same GPU the same (task, seed) aborted deterministically 5/5 times; with rendering moved to a
separate GPU the same episodes ran to completion 4/4. Xiaomi's evaluator (same layout) hit the same aborts.
Two rendering clients on one GPU abort just the same (36 aborts in 8 episodes), so the rule is **one EGL
client per GPU and nothing else on it** — or no EGL at all (`RENDER_BACKEND=osmesa`, 0 aborts in 8/8
episodes on CPU). Everything below is a second line of defence, kept because `abort()` in a driver is not
something we can catch:

* `--max-episodes-per-run 1` (default): each episode runs in a fresh client process (exit 3 = more work),
  so every episode starts with a clean MuJoCo/EGL context; the policy server stays up.
* Resume from `<task>/stats.json`; `<task>/inflight.json` counts attempts of the running episode; after
  `--max-episode-attempts` (5) native crashes on the same episode it is recorded as a failure
  (`"aborted": "native_crash"`) and skipped.
* `--max-episode-seconds` (1800): a diverged simulation (arm jammed into geometry → thousands of contacts →
  ~1.3 s/step instead of 0.05) is ended as a failure (`"aborted": "time_budget"`); `non_finite_state`
  ends an episode whose state went NaN. `aborted` counts are reported per task.
* `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `MUJOCO_EGL_DEVICE_ID=<gpu>` as in the reference evaluator.

## Verified against the training data (step_015000 smoke)

* Camera orientation/framing of `gym` observations matches the LeRobot videos (no flip needed).
* `observation.state` layout `[base_pos 3, base_quat xyzw 4, eef_pos_rel 3, eef_quat_rel xyzw 4, gripper 2]`
  matches env `state.*` keys; the home pose is identical.
* Reference frames: `ref_frames/OpenDrawer/` (dataset ep0 frame 0) vs `first_obs_ep0_*.png` in a smoke run.
