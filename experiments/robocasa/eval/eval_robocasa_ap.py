#!/usr/bin/env python3
"""RoboCasa365 rollout client for the Action-as-Patch policy server.

Runs in the robocasa conda env (robocasa 1.0.1 / robosuite 1.5.2 / mujoco), no torch needed.
Structure follows the official-protocol evaluator (task sets from robocasa's TASK_SET_REGISTRY,
horizon from get_task_horizon, seeded resets, per-task stats.json + summary). The 12-d action
returned by the server is in the LeRobot dataset layout; we build the env's dict action by
NAME from modality.json (base_motion / control_mode / end_effector_* / gripper_close), so the
flat-vector order difference vs robocasa.utils.env_utils.convert_action does not matter.

Example (single worker):
  MUJOCO_GL=egl python eval_robocasa_ap.py --server-port 20000 --task-name OpenDrawer \
      --num-trials 2 --save-videos --dump-first-obs --modality-json <dataset>/.../meta/modality.json
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image

AUTHKEY = b"ap-robocasa"
CAMERA_KEYS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def pack(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), tuple(arr.shape), arr.dtype.str


def unpack(item) -> np.ndarray:
    buf, shape, dtype = item
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)


class Modality:
    """State/action slice layout from a LeRobot meta/modality.json."""

    def __init__(self, path: Path) -> None:
        m = json.loads(Path(path).read_text())
        self.state = sorted(((k, v["start"], v["end"]) for k, v in m["state"].items()), key=lambda x: x[1])
        self.action = sorted(((k, v["start"], v["end"]) for k, v in m["action"].items()), key=lambda x: x[1])
        self.state_dim = self.state[-1][2]
        self.action_dim = self.action[-1][2]

    def obs_to_state(self, obs: dict[str, Any]) -> np.ndarray:
        parts = []
        for name, s, e in self.state:
            v = np.asarray(obs[f"state.{name}"], dtype=np.float32).reshape(-1)
            if v.shape[0] != e - s:
                raise ValueError(f"state.{name}: env gives {v.shape[0]} dims, modality.json expects {e - s}")
            parts.append(v)
        return np.concatenate(parts).astype(np.float32)

    def action_to_env(self, a: np.ndarray) -> dict[str, np.ndarray]:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.shape[0] != self.action_dim:
            raise ValueError(f"action dim {a.shape[0]} != {self.action_dim}")
        return {f"action.{name}": a[s:e].copy() for name, s, e in self.action}


class EpisodeBudgetReached(Exception):
    """Raised when --max-episodes-per-run episodes have been run in this process (exit code 3 = more work left)."""


class APClient:
    def __init__(self, host: str, port: int, connect_timeout: float = 600.0) -> None:
        deadline = time.time() + connect_timeout
        while True:
            try:
                self.conn = Client((host, port), family="AF_INET", authkey=AUTHKEY)
                break
            except (ConnectionRefusedError, OSError) as e:
                if time.time() > deadline:
                    raise RuntimeError(f"policy server {host}:{port} not reachable: {e}") from e
                time.sleep(3)
        self.conn.send({"type": "ping"})
        info = self.conn.recv()
        if info.get("type") != "pong":
            raise RuntimeError(f"bad handshake: {info}")
        self.action_horizon = int(info["action_horizon"])
        self.image_keys = list(info["image_keys"])
        self.state_dim = int(info["state_dim"])
        logging.info("Connected to %s:%d horizon=%d cams=%s", host, port, self.action_horizon, self.image_keys)

    def infer(self, images: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> tuple[np.ndarray, float]:
        self.conn.send({"type": "act",
                        "images": {k: pack(images[k]) for k in self.image_keys},
                        "state": pack(np.asarray(state, dtype=np.float32)),
                        "instruction": str(instruction)})
        reply = self.conn.recv()
        if reply.get("type") != "action":
            raise RuntimeError(f"server error: {reply}")
        return unpack(reply["action"]).astype(np.float32), float(reply.get("elapsed", 0.0))

    def close(self) -> None:
        try:
            self.conn.send({"type": "close"})
            self.conn.recv()
        except Exception:
            pass
        self.conn.close()


def collect_images(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(obs[f"video.{k}"], dtype=np.uint8) for k in CAMERA_KEYS}


def video_frame(obs: dict[str, Any]) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([obs[f"video.{k}"] for k in CAMERA_KEYS], axis=1))


def evaluate_task(env_name: str, task_index: int, args: argparse.Namespace, client: APClient,
                  modality: Modality, gym: Any, get_task_horizon: Any, output_dir: Path) -> dict[str, Any]:
    task_dir = output_dir / env_name
    task_dir.mkdir(parents=True, exist_ok=True)
    horizon = args.horizon if args.horizon is not None else int(get_task_horizon(env_name))
    results: list[dict[str, Any]] = []
    if args.resume and (task_dir / "stats.json").exists():
        try:
            results = [r for r in json.loads((task_dir / "stats.json").read_text())["episodes"] if r["episode"] < args.num_trials]
        except Exception as e:  # corrupt partial file: start over
            logging.warning("could not resume from %s: %s", task_dir / "stats.json", e)
            results = []
        if results:
            logging.info("RESUME task=%s: %d episodes already done %s", env_name, len(results), sorted(r["episode"] for r in results))
    done_episodes = {r["episode"] for r in results}
    if len(done_episodes) >= args.num_trials:
        logging.info("TASK_COMPLETE task=%s (%d episodes)", env_name, len(done_episodes))
        return _stats(env_name, args.split, horizon, results)
    inflight_path = task_dir / "inflight.json"  # which episode is running + attempt count (survives native crashes)
    logging.info("ENV_CREATE task=%s split=%s", env_name, args.split)
    t_env = time.time()
    env = gym.make(f"robocasa/{env_name}", split=args.split, seed=args.seed)
    logging.info("ENV_READY task=%s (%.1fs)", env_name, time.time() - t_env)

    def _write_stats() -> None:
        results.sort(key=lambda r: r["episode"])
        (task_dir / "stats.json").write_text(json.dumps(_stats(env_name, args.split, horizon, results), indent=2))

    try:
        for episode in range(args.num_trials):
            if episode in done_episodes:
                continue
            global_index = task_index * args.num_trials + episode
            episode_seed = args.seed + global_index
            attempt = 1
            try:
                inflight = json.loads(inflight_path.read_text())
                if inflight.get("episode") == episode:
                    attempt = int(inflight.get("attempt", 0)) + 1
            except Exception:
                pass
            if attempt > args.max_episode_attempts:
                logging.warning("SKIP task=%s ep=%d: crashed natively %d times -> recorded as failure", env_name, episode, attempt - 1)
                results.append({"episode": episode, "global_episode_index": global_index, "seed": episode_seed, "success": False,
                                "steps": 0, "wall_s": 0.0, "infer_calls": 0, "infer_s_per_call": 0.0, "aborted": "native_crash"})
                inflight_path.unlink(missing_ok=True)
                _write_stats()
                args._episodes_run += 1
                if args._episodes_run >= args.max_episodes_per_run:
                    raise EpisodeBudgetReached()
                continue
            inflight_path.write_text(json.dumps({"episode": episode, "attempt": attempt}))
            obs, _ = env.reset(seed=episode_seed)
            instruction = str(obs["annotation.human.task_description"])
            logging.info("EPISODE_START task=%s ep=%d seed=%d horizon=%d instr=%r", env_name, episode, episode_seed, horizon, instruction)
            if args.dump_first_obs and episode == 0:
                for k in CAMERA_KEYS:
                    Image.fromarray(np.asarray(obs[f"video.{k}"], dtype=np.uint8)).save(task_dir / f"first_obs_ep{episode}_{k}.png")
                np.save(task_dir / f"first_state_ep{episode}.npy", modality.obs_to_state(obs))
                logging.info("Dumped first observation PNGs + state to %s", task_dir)

            plan: collections.deque[np.ndarray] = collections.deque()
            frames = [video_frame(obs)] if (args.save_videos or args.save_failure_videos) else []
            success, steps, n_infer, infer_time, aborted = False, 0, 0, 0.0, None
            t_ep = time.time()
            state = modality.obs_to_state(obs)
            while steps < horizon:
                if not plan:
                    chunk, dt = client.infer(collect_images(obs), state, instruction)
                    n_infer += 1
                    infer_time += dt
                    if len(chunk) < args.replan_steps:
                        raise RuntimeError(f"server returned {len(chunk)} actions < replan_steps={args.replan_steps}")
                    plan.extend(chunk[: args.replan_steps])
                action = plan.popleft()
                obs, _, done, truncated, info = env.step(modality.action_to_env(action))
                steps += 1
                success = bool(info.get("success", False))
                state = modality.obs_to_state(obs)
                if not np.all(np.isfinite(state)):
                    aborted = "non_finite_state"
                    logging.warning("ABORT task=%s ep=%d step=%d: non-finite state (sim diverged)", env_name, episode, steps)
                    break
                if time.time() - t_ep > args.max_episode_seconds:
                    aborted = "time_budget"
                    logging.warning("ABORT task=%s ep=%d step=%d: exceeded --max-episode-seconds=%.0f (%.2fs/step)",
                                    env_name, episode, steps, args.max_episode_seconds, (time.time() - t_ep) / steps)
                    break
                if frames and (steps % args.video_stride == 0 or success or done or truncated):
                    frames.append(video_frame(obs))
                if steps == 1 or steps % 50 == 0:
                    try:
                        ncon = int(env.unwrapped.sim.data.ncon)
                    except Exception:
                        ncon = -1
                    logging.info("STEP task=%s ep=%d step=%d/%d infer=%d (%.2fs/infer) %.3fs/step ncon=%d",
                                 env_name, episode, steps, horizon, n_infer, infer_time / max(n_infer, 1), (time.time() - t_ep) / steps, ncon)
                if success or done or truncated:
                    break
            status = "success" if success else "failure"
            wall = time.time() - t_ep
            if frames and (args.save_videos or (args.save_failure_videos and not success)):
                path = task_dir / f"episode_{episode:03d}_seed_{episode_seed}_{status}.mp4"
                imageio.mimsave(path, frames, fps=args.video_fps)
                logging.info("VIDEO %s (%d frames)", path, len(frames))
            results.append({"episode": episode, "global_episode_index": global_index, "seed": episode_seed,
                            "success": success, "steps": steps, "wall_s": round(wall, 1),
                            "infer_calls": n_infer, "infer_s_per_call": round(infer_time / max(n_infer, 1), 3),
                            "aborted": aborted})
            logging.info("EPISODE_END task=%s ep=%d success=%s steps=%d wall=%.0fs aborted=%s", env_name, episode, success, steps, wall, aborted)
            inflight_path.unlink(missing_ok=True)
            _write_stats()
            args._episodes_run += 1
            if args._episodes_run >= args.max_episodes_per_run:
                raise EpisodeBudgetReached()
        return _stats(env_name, args.split, horizon, results)
    finally:
        env.close()


def _stats(env_name: str, split: str, horizon: int, results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    s = sum(int(r["success"]) for r in results)
    return {"env_name": env_name, "split": split, "num_episodes": n, "successes": s,
            "success_rate": (s / n) if n else 0.0, "horizon": horizon,
            "aborted": sum(1 for r in results if r.get("aborted")), "episodes": results}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an Action-as-Patch checkpoint on RoboCasa365.")
    p.add_argument("--server-addr", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=20000)
    p.add_argument("--modality-json", required=True, help="a LeRobot meta/modality.json of the training data")
    p.add_argument("--split", choices=("pretrain", "target"), default="pretrain",
                   help="object-instance split sampled by the env (pretrain = training instances)")
    p.add_argument("--task-set", default="target50", help="key of robocasa TASK_SET_REGISTRY")
    p.add_argument("--task-name", action="append", default=None, help="restrict to these tasks (repeatable)")
    p.add_argument("--num-trials", type=int, default=50)
    p.add_argument("--replan-steps", type=int, default=16)
    p.add_argument("--horizon", type=int, default=None, help="override the registry horizon")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-root-dir", default="eval_results/robocasa365")
    p.add_argument("--run-id", default=None)
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--save-failure-videos", action="store_true")
    p.add_argument("--video-stride", type=int, default=2)
    p.add_argument("--video-fps", type=int, default=20)
    p.add_argument("--dump-first-obs", action="store_true", help="save first-observation PNGs (orientation check)")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="do not skip episodes already in <task>/stats.json")
    p.add_argument("--max-episode-seconds", type=float, default=1800.0,
                   help="end an episode as failure after this wall time (guards against pathological slow sim states; normal episodes take ~40-60 s)")
    p.add_argument("--max-episode-attempts", type=int, default=5,
                   help="after this many native crashes on the same episode, record it as failed and move on")
    p.add_argument("--max-episodes-per-run", type=int, default=1,
                   help="exit with code 3 after this many episodes so the launcher restarts a fresh process "
                        "(fresh MuJoCo/EGL context per episode avoids reset-related native aborts); 0 = unlimited")
    return p.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("--worker-index must be in [0, --num-workers)")
    import gymnasium as gym
    import robocasa  # noqa: F401  (registers robocasa/* envs)
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if args.task_set not in TASK_SET_REGISTRY:
        raise KeyError(f"unknown task set {args.task_set!r}; available: {sorted(TASK_SET_REGISTRY)}")
    all_tasks = list(TASK_SET_REGISTRY[args.task_set])
    task_to_index = {t: i for i, t in enumerate(all_tasks)}
    selected = list(args.task_name) if args.task_name else all_tasks
    unknown = [t for t in selected if t not in task_to_index]
    if unknown:
        raise KeyError(f"tasks not in {args.task_set!r}: {unknown}")
    mine = selected[args.worker_index :: args.num_workers]

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.save_root_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("worker %d/%d tasks=%s -> %s", args.worker_index, args.num_workers, mine, output_dir)

    modality = Modality(Path(args.modality_json))
    args._episodes_run = 0
    if args.max_episodes_per_run <= 0:
        args.max_episodes_per_run = 10**9
    client = APClient(args.server_addr, args.server_port)
    task_stats: dict[str, Any] = {}
    try:
        for env_name in mine:
            task_stats[env_name] = evaluate_task(env_name, task_to_index[env_name], args, client, modality, gym, get_task_horizon, output_dir)
    except EpisodeBudgetReached:
        logging.info("EPISODE_BUDGET: ran %d episode(s) in this process, more work left -> exit 3", args._episodes_run)
        client.close()
        raise SystemExit(3)
    finally:
        client.close()

    n_ep = sum(s["num_episodes"] for s in task_stats.values())
    n_ok = sum(s["successes"] for s in task_stats.values())
    summary = {"split": args.split, "task_set": args.task_set, "worker_index": args.worker_index, "num_workers": args.num_workers,
               "replan_steps": args.replan_steps, "seed": args.seed, "num_tasks": len(task_stats), "num_episodes": n_ep,
               "successes": n_ok, "episode_success_rate": (n_ok / n_ep) if n_ep else None,
               "mean_task_success_rate": float(np.mean([s["success_rate"] for s in task_stats.values()])) if task_stats else None,
               "tasks": task_stats}
    (output_dir / f"summary_worker{args.worker_index}.json").write_text(json.dumps(summary, indent=2))
    logging.info("DONE worker %d: %d/%d episodes succeeded (%.1f%%)", args.worker_index, n_ok, n_ep, 100.0 * (n_ok / n_ep if n_ep else 0))


if __name__ == "__main__":
    main()
