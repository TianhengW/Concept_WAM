#!/usr/bin/env python3
"""Action-as-Patch policy server for RoboCasa365 evaluation.

Runs inside the AP training venv on one GPU. The RoboCasa simulator client
(experiments/robocasa/eval/eval_robocasa_ap.py, robocasa conda env) connects over
multiprocessing.connection and sends {images, state, instruction}; we reply with a
denormalized action chunk [T, 12] in the LeRobot dataset layout (modality.json order).

Numpy arrays cross the wire as (bytes, shape, dtype) so the two sides may run different
numpy major versions.

Preprocessing mirrors training exactly (RobotVideoDataset + ImageWAMProcessor):
  per-cam uint8 HxWx3 -> [0,1] CHW -> Resize(shape_meta) -> "robotwin" compact concat
  (top=cam0 192x256, bottom=cam1|cam2 96x128 each) -> (x-0.5)/0.5 -> [1,3,288,256]
  proprio 16d -> processor.action_state_transform -> normalizer.forward
  prompt = DEFAULT_PROMPT.format(task=instruction)
"""
from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(os.environ["REPO_ROOT"]) if os.environ.get("REPO_ROOT") else Path(__file__).resolve().parents[3]
for _p in (REPO_ROOT / "src", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from imagewam.datasets.lerobot.processors.imagewam_processor import ImageWAMProcessor  # noqa: E402
from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset  # noqa: E402
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402

AUTHKEY = b"ap-robocasa"


def pack(arr: np.ndarray) -> tuple[bytes, tuple[int, ...], str]:
    arr = np.ascontiguousarray(arr)
    return arr.tobytes(), tuple(arr.shape), arr.dtype.str


def unpack(item) -> np.ndarray:
    buf, shape, dtype = item
    return np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)


def _resolve_target(target: str):
    import importlib
    module_name, _, attr_name = target.rpartition(".")
    return getattr(importlib.import_module(module_name), attr_name)


def _filter_model_cfg_for_target(model_cfg: DictConfig) -> DictConfig:
    """Drop keys the factory does not accept (same as experiments/libero/eval_libero_single.py)."""
    model_dict = OmegaConf.to_container(model_cfg, resolve=True)
    target = model_dict.get("_target_")
    if not target:
        return OmegaConf.create(model_dict)
    signature = inspect.signature(_resolve_target(str(target)))
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return OmegaConf.create(model_dict)
    allowed = {"_target_"} | {
        n for n, p in signature.parameters.items()
        if p.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    dropped = sorted(set(model_dict) - allowed)
    if dropped:
        logging.info("Ignoring unsupported model config keys for %s: %s", target, dropped)
    return OmegaConf.create({k: v for k, v in model_dict.items() if k in allowed})


class APPolicy:
    def __init__(self, run_dir: Path, ckpt: Path, stats: Path, device: str,
                 num_inference_steps: int, seed: int | None) -> None:
        cfg = OmegaConf.load(run_dir / "config.yaml")
        self.cfg = cfg
        self.device = device
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed

        model_cfg = _filter_model_cfg_for_target(cfg.model)
        t0 = time.time()
        self.model = instantiate(model_cfg, model_dtype=torch.bfloat16, device=device)
        self.model.load_checkpoint(str(ckpt))
        self.model = self.model.to(device).eval()
        logging.info("Model loaded from %s in %.1fs", ckpt, time.time() - t0)

        self.processor: ImageWAMProcessor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
        logging.info("Dataset stats: %s", stats)

        data_cfg = cfg.data.train
        images_meta = data_cfg.shape_meta["images"]
        self.image_keys = [str(m["key"]) for m in images_meta]
        self.cam_hw = [(int(m["shape"][1]), int(m["shape"][2])) for m in images_meta]
        if str(data_cfg.get("concat_multi_camera")) != "robotwin":
            raise ValueError("This server implements the 'robotwin' compact camera layout only.")
        self.top, self.left, self.right = RobotVideoDataset._robotwin_camera_sizes(str(data_cfg.robotwin_camera_layout))
        self.video_hw = (int(data_cfg.video_size[0]), int(data_cfg.video_size[1]))
        self.action_horizon = int(data_cfg.num_frames) - 1
        self.action_key = str(data_cfg.shape_meta["action"][0]["key"])
        self.state_key = str(data_cfg.shape_meta["state"][0]["key"])
        self.state_dim = int(data_cfg.shape_meta["state"][0]["shape"])
        self.infer_params = set(inspect.signature(self.model.infer_action).parameters)
        logging.info("cams=%s layout top=%s left=%s right=%s video=%s horizon=%d",
                     self.image_keys, self.top, self.left, self.right, self.video_hw, self.action_horizon)

    def preprocess_images(self, images: dict[str, np.ndarray]) -> torch.Tensor:
        cams = []
        for key, (h, w) in zip(self.image_keys, self.cam_hw):
            img = images[key]
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"{key}: expected HxWx3 uint8, got {img.shape} {img.dtype}")
            x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0  # ToTensor
            if tuple(x.shape[1:]) != (h, w):  # processor train_transforms: Resize(shape_meta)
                x = TF.resize(x, [h, w], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            cams.append(x)
        top = TF.resize(cams[0], list(self.top), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        left = TF.resize(cams[1], list(self.left), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        right = TF.resize(cams[2], list(self.right), interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        x = torch.cat([top, torch.cat([left, right], dim=-1)], dim=-2)  # [3, 288, 256]
        if tuple(x.shape[1:]) != self.video_hw:
            raise ValueError(f"concat gave {tuple(x.shape[1:])}, expected {self.video_hw}")
        x = (x - 0.5) / 0.5  # Normalize(mean=0.5, std=0.5) -> [-1, 1]
        return x.unsqueeze(0).to(device=self.device, dtype=self.model.torch_dtype)

    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state dim {state.shape[0]} != {self.state_dim}")
        batch = {"state": {self.state_key: torch.as_tensor(state).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        return batch["state"][self.state_key]

    @torch.no_grad()
    def act(self, images: dict[str, np.ndarray], state: np.ndarray, instruction: str,
            action_horizon: int | None = None) -> np.ndarray:
        kwargs = dict(
            prompt=DEFAULT_PROMPT.format(task=instruction),
            input_image=self.preprocess_images(images),
            action_horizon=int(action_horizon or self.action_horizon),
            proprio=self.normalize_state(state),
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
            rand_device="cpu",
        )
        kwargs = {k: v for k, v in kwargs.items() if k in self.infer_params}
        pred = self.model.infer_action(**kwargs)
        action = pred["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"][self.action_key]
        action = normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()[0]
        return np.asarray(action, dtype=np.float32)  # [T, 12], dataset (modality.json) layout


def serve(policy: APPolicy, host: str, port: int, ready_file: Path | None) -> None:
    listener = Listener((host, port), family="AF_INET", authkey=AUTHKEY)
    logging.info("Serving on %s:%d", host, port)
    if ready_file is not None:
        ready_file.parent.mkdir(parents=True, exist_ok=True)
        ready_file.write_text(f"{host}:{port}\n")
    n_requests = 0
    while True:
        conn = listener.accept()
        logging.info("Client connected")
        try:
            while True:
                msg = conn.recv()
                kind = msg.get("type")
                if kind == "ping":
                    conn.send({"type": "pong", "action_horizon": policy.action_horizon,
                               "image_keys": policy.image_keys, "state_dim": policy.state_dim})
                elif kind == "act":
                    t0 = time.perf_counter()
                    images = {k: unpack(v) for k, v in msg["images"].items()}
                    action = policy.act(images, unpack(msg["state"]), str(msg["instruction"]),
                                        msg.get("action_horizon"))
                    dt = time.perf_counter() - t0
                    n_requests += 1
                    if n_requests % 20 == 1:
                        logging.info("act #%d: %.3fs  action[0]=%s", n_requests, dt, np.round(action[0], 3).tolist())
                    conn.send({"type": "action", "action": pack(action), "elapsed": dt})
                elif kind == "close":
                    conn.send({"type": "bye"})
                    break
                else:
                    conn.send({"type": "error", "message": f"unknown request type {kind!r}"})
        except EOFError:
            logging.info("Client disconnected")
        except Exception:  # keep serving other clients; report the failure
            logging.exception("Request failed")
            try:
                conn.send({"type": "error", "message": "server exception; see server log"})
            except Exception:
                pass
        finally:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="training run dir with config.yaml (+ dataset_stats.json)")
    parser.add_argument("--ckpt", required=True, help="checkpoints/weights/step_XXXXXX.pt")
    parser.add_argument("--stats", default=None, help="dataset_stats.json (default: <run-dir>/dataset_stats.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ready-file", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    stats = Path(args.stats) if args.stats else run_dir / "dataset_stats.json"
    policy = APPolicy(run_dir, Path(args.ckpt), stats, args.device, args.num_inference_steps, args.seed)
    serve(policy, args.host, args.port, Path(args.ready_file) if args.ready_file else None)


if __name__ == "__main__":
    main()
