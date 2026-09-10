#!/usr/bin/env python3
"""Compute ImageWAM normalization stats (dataset_stats.json) for a RoboDojo LeRobot root.

Instantiates the training dataset exactly as `task=robodojo_flux2_klein_4b_actionpatch_full`
does (same processor / shape_meta / nonidle filter), but with pretrained_norm_stats=null so
RobotVideoDataset computes global mean/std over the (filtered) training set; the result is
then saved next to the dataset so every later run (train, eval) reads the same file.

Usage:
  .venv/bin/python scripts/data/compute_robodojo_dataset_stats.py \
      --dataset-dir /storage/yukaichengLab/share/datasets/RoboDojo_lerobot \
      [--nonidle /path/nonidle_ranges.json] [--task robodojo_flux2_klein_4b_actionpatch_full] \
      [--out dataset_stats.json]
"""
import argparse
import logging
import os
import sys

REPO = "/storage/yukaichengLab/mazijian/wth/ImageWAM"
for p in (REPO, os.path.join(REPO, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from imagewam.datasets.lerobot.utils.normalizer import save_dataset_stats_to_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, nargs="+", help="one or more LeRobot roots (stats pooled over all)")
    ap.add_argument("--nonidle", default=None, help="single nonidle json (single root) — default: <root>/nonidle_ranges.json per root if present")
    ap.add_argument("--task", default="robodojo_flux2_klein_4b_actionpatch_full")
    ap.add_argument("--out", default=None, help="default: <first dataset-dir>/dataset_stats.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    roots = [os.path.abspath(d) for d in args.dataset_dir]
    out = args.out or os.path.join(roots[0], "dataset_stats.json")
    if args.nonidle is not None:
        nonidle_cfg = args.nonidle if args.nonidle != "null" else None
    else:
        per_root = {r: os.path.join(r, "nonidle_ranges.json") for r in roots if os.path.exists(os.path.join(r, "nonidle_ranges.json"))}
        if len(per_root) == 0:
            nonidle_cfg = None
        elif len(per_root) == len(roots):
            nonidle_cfg = per_root  # per-root dict, as MultiLeRobotDataset expects
        else:
            raise SystemExit(f"nonidle_ranges.json present for only {list(per_root)} of {roots}; generate all or pass --nonidle")

    # Hydra's override grammar cannot express dict keys containing '/', so compose the task
    # config and then patch dataset_dirs / nonidle / stats programmatically.
    initialize_config_dir(config_dir=os.path.join(REPO, "configs"), version_base="1.3")
    cfg = compose(config_name="train", overrides=[f"task={args.task}"])
    OmegaConf.set_struct(cfg, False)
    cfg.data.train.dataset_dirs = roots
    cfg.data.train.nonidle_filter_path = nonidle_cfg
    cfg.data.train.pretrained_norm_stats = None
    print("dataset_dirs:", list(cfg.data.train.dataset_dirs))
    print("nonidle:", cfg.data.train.nonidle_filter_path)

    # Build the dataset without a processor so nothing is normalized yet, then run the
    # same stats routine the trainer uses on rank 0.
    train_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train, resolve=True))
    processor_cfg = train_cfg.pop("processor")
    ds = instantiate(train_cfg, processor=None)
    processor = instantiate(processor_cfg)
    print("train dataset len (samples):", len(ds))
    stats = ds.lerobot_dataset.get_dataset_stats(processor)
    save_dataset_stats_to_json(stats, out)
    print("saved", out)
    for group in ("state", "action"):
        for key, val in stats[group].items():
            gm = val.get("global_mean")
            gs = val.get("global_std")
            print(f"{group}/{key}: mean[:4]={[round(float(x), 4) for x in list(gm)[:4]]} std[:4]={[round(float(x), 4) for x in list(gs)[:4]]}")
    print("num_episodes", stats.get("num_episodes"), "num_transition", stats.get("num_transition"))


if __name__ == "__main__":
    main()
