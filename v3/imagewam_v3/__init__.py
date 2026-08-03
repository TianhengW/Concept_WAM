"""ImageWAM v3 — VL LatentReasoner (image-grounded Coconut-style conditioning).

v3 replaces the text-only Qwen3-4B reasoner of v2_1 with Qwen3-VL-4B-Instruct so the
initial observation frame ``t0`` is fed into the LLM (through the vision tower) and the
latent reasoning trajectory is grounded in the actual scene. The K reasoning tokens are
handed to the MoT as the whole text conditioning (Plan A). Output layout on the MoT
``txt_in`` side is unchanged (``[B, K, 7680]`` + mask).

Lives under ImageWAM/v3/ and imports the base package (imagewam.*) without editing it.
Requires transformers >= 4.57 (Qwen3-VL); use the isolated ``.venv_v3``.
"""

from .vl_latent_reasoner import VLLatentReasoner, FeedbackProjector
from .model_v3 import ImageWAMV3
from .runtime_v3 import create_imagewam_v3

__all__ = ["VLLatentReasoner", "FeedbackProjector", "ImageWAMV3", "create_imagewam_v3"]
