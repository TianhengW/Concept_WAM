#!/usr/bin/env python3
"""Step a RoboCasa365 env with policy-free actions to see whether the native abort is env-intrinsic.
usage: probe_env.py <task> <seed> <mode: zero|random> [horizon]
Logs ncon / sim time / finite-state every 25 steps; writes progress to stdout (flushed) so the crash step is exact.
"""
import sys, time, json
import numpy as np

task, seed, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
horizon = int(sys.argv[4]) if len(sys.argv) > 4 else 900
import gymnasium as gym
import robocasa  # noqa
from robocasa.utils.dataset_registry_utils import get_task_horizon

rng = np.random.default_rng(0)
env = gym.make(f"robocasa/{task}", split="pretrain", seed=7)
obs, _ = env.reset(seed=seed)
print(f"[probe] task={task} seed={seed} mode={mode} horizon={horizon} registry_horizon={get_task_horizon(task)} instr={obs['annotation.human.task_description']!r}", flush=True)
sim = env.unwrapped.sim
t0 = time.time()
for step in range(1, horizon + 1):
    if mode == "zero":
        a = {"action.base_motion": np.zeros(4, np.float32), "action.control_mode": np.array([-1.0], np.float32),
             "action.end_effector_position": np.zeros(3, np.float32), "action.end_effector_rotation": np.zeros(3, np.float32),
             "action.gripper_close": np.array([-1.0], np.float32)}
    else:
        a = {"action.base_motion": np.zeros(4, np.float32), "action.control_mode": np.array([-1.0], np.float32),
             "action.end_effector_position": rng.uniform(-1, 1, 3).astype(np.float32), "action.end_effector_rotation": rng.uniform(-0.3, 0.3, 3).astype(np.float32),
             "action.gripper_close": np.array([rng.choice([-1.0, 1.0])], np.float32)}
    obs, r, done, trunc, info = env.step(a)
    if step % 25 == 0 or step == 1:
        qpos_ok = bool(np.all(np.isfinite(sim.data.qpos)))
        print(f"[probe] step={step} t_sim={sim.data.time:.2f} ncon={sim.data.ncon} qpos_finite={qpos_ok} "
              f"eef={np.round(obs['state.end_effector_position_relative'],3).tolist()} {(time.time()-t0)/step:.3f}s/step done={done} succ={info.get('success')}", flush=True)
    else:
        print(f"[probe] step={step}", flush=True)
print("[probe] COMPLETED without crash", flush=True)
env.close()
