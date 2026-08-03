"""ImageWAM v2 — Arm A.

Use Qwen3-4B as a *generator* of a small, fixed set of conditioning embeddings
(K learnable query slots read out in a single forward), fed to the MoT to
generate image and action. This differs from v1:
  - v1 (LatentReasoner): K reasoning tokens appended to the full prompt sequence.
  - v2 Arm A: K query slots are the whole text conditioning (a bottleneck), and
    Qwen is fine-tuned end-to-end (full parameters, no LoRA) by the downstream
    image/action flow-matching loss.

Everything here lives under `ImageWAM/v2/` and imports the base package
(`imagewam.*`) without editing it.
"""

from .qwen_slot_generator import QwenSlotGenerator
from .model_v2 import ImageWAMV2
from .runtime_v2 import create_imagewam_v2

__all__ = ["QwenSlotGenerator", "ImageWAMV2", "create_imagewam_v2"]
