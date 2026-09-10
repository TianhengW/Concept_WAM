"""Create step_000000.pt checkpoints for base & patch = the untrained init models.

base  : pretrained FLUX video expert + RANDOM-init action DiT (seed 0)
patch : pretrained FLUX only (codec has no params)
Saved into the OLD run dirs (which hold dataset_stats.json) so the eval
scripts can point EXP_PATH there with EVAL_TRAIN_STEP=000000.
"""

import torch

FLUX2_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
AE_MODEL = "/storage/yukaichengLab/share/model/FLUX.2-dev/ae.safetensors"
FLUX2_SRC = "/storage/yukaichengLab/mazijian/wth/flux2"
REPO = "/storage/yukaichengLab/mazijian/wth/ImageWAM"
OUT_BASE = REPO + "/runs/robotwin_flux2_klein_4b_base_sub20/2026-08-06_20-46-54/checkpoints/weights/step_000000.pt"
OUT_PATCH = REPO + "/runs/robotwin_flux2_klein_4b_actionpatch_sub20/2026-08-06_22-50-33/checkpoints/weights/step_000000.pt"

torch.manual_seed(0)

from imagewam.runtime import create_imagewam_flux2_klein
m = create_imagewam_flux2_klein(
    flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
    qwen3_model_spec="/storage/yukaichengLab/share/model/Qwen3-4B", qwen_context_len=128,
    load_text_encoder=False, proprio_dim=14,
    action_dit_config={"action_dim": 14, "hidden_dim": 1024, "mlp_ratio": 4.0,
                       "max_action_horizon": 64, "use_gradient_checkpointing": False},
    action_scheduler={"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000},
    loss={"lambda_video": 0.5, "lambda_action": 1.0},
    concept_k=0, lambda_concept=0.0,
    model_dtype=torch.bfloat16, device="cuda",
)
m.save_checkpoint(OUT_BASE, step=0)
print("saved base step0:", OUT_BASE, flush=True)
del m
torch.cuda.empty_cache()

torch.manual_seed(0)
from exploration.action_img_patch.model import ImageWAMActionPatch
m = ImageWAMActionPatch.from_flux2_klein_actionpatch_pretrained(
    flux2_model_path=FLUX2_MODEL, ae_model_path=AE_MODEL, flux2_src_path=FLUX2_SRC,
    qwen3_model_spec="/storage/yukaichengLab/share/model/Qwen3-4B", qwen_context_len=128,
    proprio_dim=14, load_text_encoder=False,
    device="cuda", torch_dtype=torch.bfloat16, action_patch_dim=14,
)
m.save_checkpoint(OUT_PATCH, step=0)
print("saved patch step0:", OUT_PATCH, flush=True)
