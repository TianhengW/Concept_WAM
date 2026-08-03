"""Arm A training entry point.

Reuses the base `run_training` (datasets + Wan22Trainer) but composes a
self-contained v2 config that points `model._target_` at
`imagewam_v2.runtime_v2.create_imagewam_v2`.
"""
import hydra
from omegaconf import DictConfig

from imagewam.runtime import run_training
from imagewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="robotwin_v2_armA", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
