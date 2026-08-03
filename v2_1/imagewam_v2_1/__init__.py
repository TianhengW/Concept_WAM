"""ImageWAM v2_1 — LatentReasoner (Coconut-style) conditioning.

Copied from the Automodel checkout of ImageWAM. Unlike v2 Arm A (full-param Qwen
as a K-slot *bottleneck* generator, single forward), this uses Qwen3 frozen + LoRA
and appends K reasoning representations *after* the prompt, produced in one of two
modes:
  - mode="thought":      K learnable thought embeddings, single differentiable forward.
  - mode="latent_loop":  Coconut continuous latent reasoning — feed each step last
                         hidden state back as the next input embedding (KV cache kept),
                         K fixed steps, fully differentiable, no token sampled.

The module lives under ImageWAM/v2_1/ and imports the base package (imagewam.*)
without editing it. Wiring into imagewam.py / trainer / runtime is TODO (see the
Automodel checkout hooks for reference).
"""

from .latent_reasoner import LatentReasoner, FeedbackProjector

__all__ = ["LatentReasoner", "FeedbackProjector"]
