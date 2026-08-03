"""v3 training entry point.

Reuses the base `run_training` (datasets + trainer) but composes a self-contained
v3 config that points `model._target_` at `imagewam_v3.runtime_v3.create_imagewam_v3`.
"""
import hydra
from omegaconf import DictConfig

from imagewam.runtime import run_training
from imagewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="robotwin_v3", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
