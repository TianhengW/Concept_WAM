"""Smoke: instantiate the aug_full mixed dataset exactly as training would."""
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

R = "/storage/yukaichengLab/mazijian/wth/ImageWAM"
initialize_config_dir(config_dir=f"{R}/configs", version_base="1.3")
cfg = compose(config_name="train",
              overrides=["task=robotwin_flux2_klein_4b_actionpatch_aug_full"])
print("dataset_dirs:", list(cfg.data.train.dataset_dirs))
print("nonidle_filter_path:", dict(cfg.data.train.nonidle_filter_path))
print("hparams: bs=%s accum=%s lr=%s epochs=%s sched=%s wd=%s" % (
    cfg.batch_size, cfg.gradient_accumulation_steps, cfg.learning_rate,
    cfg.num_epochs, cfg.lr_scheduler_type, cfg.weight_decay))

ds = instantiate(cfg.data.train)
n = len(ds)
print("TRAIN dataset len:", n)

for name, i in [("head", 0), ("tail(aug 区)", n - 1), ("mid", n // 2)]:
    s = ds[i]
    desc = {}
    for k, v in s.items():
        if hasattr(v, "shape"):
            desc[k] = tuple(v.shape)
    print(f"sample[{name}] idx={i}: " + ", ".join(f"{k}={v}" for k, v in sorted(desc.items())))
print("SMOKE_OK")
