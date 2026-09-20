#!/usr/bin/env python3
"""RoboCasa365 single-task evaluation for the action-as-patch policy — SKELETON.

Mirrors the structure of experiments/libero/eval_libero_single.py, adapted to the
robocasa sim (robosuite-based). Fill in the TODOs after installing the official
environment (https://github.com/robocasa/robocasa; see leaderboard protocol).

Pipeline per episode:
  1. env = robocasa.make(task, kitchen_scene, ...)   # official task/scene sampling
  2. obs -> model input:
       - cameras: robot0_agentview_left / _right / _eye_in_hand (256x256 each)
       - IMPORTANT: match training preprocessing exactly —
         ToTensor -> per-cam Resize(256,256) -> "robotwin" compact concat 288x256
         (top = agentview_left; bottom = agentview_right | eye_in_hand).
         Check whether robosuite returns images flipped (LIBERO needed [::-1, ::-1];
         verify against a training video frame side-by-side before trusting SR!)
       - proprio: observation.state layout (16d) must match the dataset field order.
  3. policy: ImageWAMActionPatch.infer_action (joint denoise, take action tokens,
     mean-decode) -> 12d action chunk (horizon 16), replan per train config.
  4. action -> env.step: verify normalization inverse (z-score per-task stats) and
     any gripper sign convention against the dataset before executing.
  5. SR per official horizon; aggregate per split (atomic / composite-seen / -unseen).

Run one (task, split, ckpt) per process; parallelize with a manager like
experiments/libero/run_libero_manager.py.
"""

raise SystemExit(
    "Skeleton only: install the official robocasa environment and fill in the TODOs "
    "(see experiments/robocasa/README.md)."
)
