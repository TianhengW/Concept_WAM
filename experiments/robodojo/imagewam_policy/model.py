"""XPolicyLab adapter: ImageWAM action-as-patch policy for RoboDojo (arx_x5, joint control).

Reuses `experiments/robotwin/imagewam_policy/deploy_policy.py:WorldActionRobotWinPolicy`
(hydra compose of the training task config, checkpoint + dataset_stats loading, compact
288x256 three-camera tiling, z-score normalisation, joint image/action denoising), so the
sim-eval preprocessing is bit-for-bit the RoboTwin one. This file only translates between
the RoboDojo observation/action dicts and that policy's RoboTwin-style interface.

RoboDojo obs (see RoboDojo/env/observation_manager/obs_manager.py):
    obs["vision"][cam]["color"]   (480,640,3) uint8 RGB, cam in {cam_head, cam_left_wrist, cam_right_wrist}
    obs["state"]["left_arm_joint_state"] (6,) rad, ["left_ee_joint_state"] (1,) in [0,1], right_* likewise
    obs["instruction"]            str
Action dict per step (joint mode, RoboDojo/src/eval_client/eval_env.py):
    {"left_arm_joint_state": (6,), "left_ee_joint_state": (1,), "right_arm_joint_state": (6,), "right_ee_joint_state": (1,)}
14-d packing everywhere: [L_arm6, L_grip1, R_arm6, R_grip1] (XPolicyLab.utils.process_data.pack_robot_state).

Checkpoint resolution (first hit wins):
    $IMAGEWAM_CKPT env var
    model_cfg["checkpoint_path"] (deploy.yml / --overrides)
    ckpt_name that is an existing .pt path
    <policy_dir>/checkpoints/<ckpt_name>/*.pt   (XPolicyLab convention; symlink your run's step_XXXXXX.pt there)
dataset_stats.json: $IMAGEWAM_STATS > model_cfg["dataset_stats_path"] > next to the ckpt dir > runs/<name>/<ts>/dataset_stats.json
(two levels above checkpoints/weights/step_X.pt).
"""
from __future__ import annotations

import glob
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from XPolicyLab.model_template import ModelTemplate

logger = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).resolve().parent
# experiments/robodojo/imagewam_policy -> ImageWAM repo root (works through the
# XPolicyLab/policy/imagewam_policy symlink because Path.resolve() follows it).
IMAGEWAM_ROOT = POLICY_DIR.parents[2]
ROBOTWIN_DEPLOY = IMAGEWAM_ROOT / "experiments" / "robotwin" / "imagewam_policy" / "deploy_policy.py"

CAM_HEAD, CAM_LEFT, CAM_RIGHT = "cam_head", "cam_left_wrist", "cam_right_wrist"
ARM_DIM, GRIP_DIM = 6, 1


def _load_robotwin_deploy_module():
    spec = importlib.util.spec_from_file_location("imagewam_robotwin_deploy_policy", str(ROBOTWIN_DEPLOY))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _none_like(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in {"", "none", "null"})


def _resolve_checkpoint(model_cfg: Dict[str, Any]) -> Path:
    env = os.environ.get("IMAGEWAM_CKPT")
    if not _none_like(env):
        return Path(env).expanduser().resolve()
    cfg_path = model_cfg.get("checkpoint_path")
    if not _none_like(cfg_path):
        return Path(str(cfg_path)).expanduser().resolve()
    ckpt_name = str(model_cfg.get("ckpt_name") or "")
    if ckpt_name and os.path.isfile(ckpt_name):
        return Path(ckpt_name).resolve()
    if ckpt_name:
        cand_dir = POLICY_DIR / "checkpoints" / ckpt_name
        pts = sorted(glob.glob(str(cand_dir / "*.pt")))
        if len(pts) == 1:
            return Path(pts[0]).resolve()
        if len(pts) > 1:
            raise ValueError(f"{cand_dir} holds {len(pts)} .pt files; keep exactly one or set IMAGEWAM_CKPT")
    raise FileNotFoundError(
        "No ImageWAM checkpoint: set IMAGEWAM_CKPT, deploy.yml checkpoint_path, pass a .pt path as "
        f"ckpt_name, or create {POLICY_DIR}/checkpoints/<ckpt_name>/<step>.pt"
    )


def _resolve_stats(model_cfg: Dict[str, Any], ckpt: Path) -> Path:
    env = os.environ.get("IMAGEWAM_STATS")
    if not _none_like(env):
        return Path(env).expanduser().resolve()
    cfg_path = model_cfg.get("dataset_stats_path")
    if not _none_like(cfg_path):
        return Path(str(cfg_path)).expanduser().resolve()
    # checkpoints/<ckpt_name>/dataset_stats.json  or  runs/<name>/<ts>/dataset_stats.json
    for cand in (ckpt.parent / "dataset_stats.json", ckpt.parents[2] / "dataset_stats.json" if len(ckpt.parents) > 2 else None):
        if cand is not None and cand.exists():
            return cand.resolve()
    raise FileNotFoundError(f"dataset_stats.json not found next to {ckpt}; set IMAGEWAM_STATS or deploy.yml dataset_stats_path")


class Model(ModelTemplate):
    def __init__(self, model_cfg: Dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg.get("action_type") or "joint")
        if self.action_type != "joint":
            raise NotImplementedError(
                f"imagewam_policy predicts joint targets; got action_type={self.action_type!r}. "
                "Run robodojo.sh with --action-type joint."
            )
        self.env_cfg_type = str(model_cfg.get("env_cfg_type") or "arx_x5")

        ckpt = _resolve_checkpoint(self.model_cfg)
        stats = _resolve_stats(self.model_cfg, ckpt)
        self.replan_steps = int(model_cfg.get("replan_steps") or 16)
        seed_raw = model_cfg.get("seed")
        infer_seed = None if _none_like(seed_raw) else int(seed_raw)

        deploy = _load_robotwin_deploy_module()
        usr_args = {
            "sim_cfg_name": model_cfg.get("sim_cfg_name") or "sim_robotwin.yaml",
            "sim_task": model_cfg.get("imagewam_task") or "robodojo_flux2_klein_4b_actionpatch_full",
            "ckpt_setting": str(ckpt),
            "dataset_stats_path": str(stats),
            "device": model_cfg.get("device") or "cuda",
            "mixed_precision": model_cfg.get("mixed_precision") or "bf16",
            "action_horizon": model_cfg.get("action_horizon"),
            "replan_steps": self.replan_steps,
            "num_inference_steps": model_cfg.get("num_inference_steps"),
            "sigma_shift": model_cfg.get("sigma_shift"),
            "seed": infer_seed,
            "text_cfg_scale": model_cfg.get("text_cfg_scale", 1.0),
            "negative_prompt": model_cfg.get("negative_prompt", ""),
            "rand_device": model_cfg.get("rand_device", "cpu"),
            "tiled": model_cfg.get("tiled", False),
            "timing_enabled": model_cfg.get("timing_enabled", False),
            "robotwin_camera_layout": model_cfg.get("robotwin_camera_layout", "compact_288x256"),
            "model_overrides": model_cfg.get("model_overrides") or None,
        }
        self.policy = deploy.get_model(usr_args)
        self._obs: Optional[Dict[str, Any]] = None
        self._obs_batch: Dict[int, Dict[str, Any]] = {}
        logger.info(
            "[imagewam_policy] ckpt=%s stats=%s task=%s horizon=%d replan=%d",
            ckpt, stats, usr_args["sim_task"], self.policy.action_horizon, self.policy.replan_steps,
        )
        print(f"[Model] imagewam_policy ready | ckpt={ckpt} | replan={self.policy.replan_steps}", flush=True)

    # ------------------------------------------------------------------ obs / action glue
    @staticmethod
    def _pack_state(state: Dict[str, Any]) -> np.ndarray:
        parts = [
            np.asarray(state["left_arm_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["left_ee_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["right_arm_joint_state"], dtype=np.float32).reshape(-1),
            np.asarray(state["right_ee_joint_state"], dtype=np.float32).reshape(-1),
        ]
        vec = np.concatenate(parts)
        if vec.shape[0] != 2 * (ARM_DIM + GRIP_DIM):
            raise ValueError(f"packed state has {vec.shape[0]} dims, expected 14")
        return vec

    @staticmethod
    def _to_robotwin_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
        vision = obs["vision"]

        def rgb(cam: str) -> np.ndarray:
            arr = np.asarray(vision[cam]["color"])
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"{cam} color has shape {arr.shape}, expected (H,W,3)")
            return arr.astype(np.uint8, copy=False)

        return {
            "observation": {
                "head_camera": {"rgb": rgb(CAM_HEAD)},
                "left_camera": {"rgb": rgb(CAM_LEFT)},
                "right_camera": {"rgb": rgb(CAM_RIGHT)},
            },
            "joint_action": {"vector": Model._pack_state(obs["state"])},
        }

    @staticmethod
    def _unpack_action(vec: np.ndarray) -> Dict[str, np.ndarray]:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        l_arm, l_grip = vec[0:ARM_DIM], vec[ARM_DIM:ARM_DIM + GRIP_DIM]
        r_arm, r_grip = vec[7:7 + ARM_DIM], vec[7 + ARM_DIM:7 + ARM_DIM + GRIP_DIM]
        return {
            "left_arm_joint_state": l_arm.copy(),
            "left_ee_joint_state": np.clip(l_grip, 0.0, 1.0),
            "right_arm_joint_state": r_arm.copy(),
            "right_ee_joint_state": np.clip(r_grip, 0.0, 1.0),
        }

    def _predict_chunk(self, obs: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
        instruction = str(obs.get("instruction") or "")
        chunk = self.policy._infer_action_chunk(self._to_robotwin_obs(obs), instruction)  # [T,14]
        n_exec = min(self.policy.replan_steps, chunk.shape[0])
        return [self._unpack_action(chunk[i]) for i in range(n_exec)]

    # ------------------------------------------------------------------ XPolicyLab contract
    def update_obs(self, obs):
        self._obs = obs

    def update_obs_batch(self, obs_list):
        self._obs_batch = {}
        for i, obs in enumerate(obs_list):
            self._obs_batch[int(obs.get("env_idx", i))] = obs

    def get_action(self):
        if self._obs is None:
            raise RuntimeError("get_action called before update_obs")
        return self._predict_chunk(self._obs)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = sorted(self._obs_batch)
        return [self._predict_chunk(self._obs_batch[int(i)]) for i in env_idx_list]

    def reset(self):
        self._obs = None
        self._obs_batch = {}
        self.policy.reset()
