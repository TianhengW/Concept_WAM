"""Domain-randomization augmentation for clean-to-randomization (C2R).

- ``DomainRandomization``: drop-in for the data config's ``video_augmentation``
  field (A-tier photometric out of the box; B-tier background when masks given).
- ``PhotometricRandomize``: A-tier photometric randomization.
- ``MaskGuidedBackground`` / ``BackgroundBank``: B-tier background replacement.
- ``PrecomputedMaskProvider`` / ``NullMaskProvider``: foreground-mask sources.
"""
from .photometric import PhotometricRandomize
from .background import BackgroundBank, MaskGuidedBackground
from .mask_provider import NullMaskProvider, PrecomputedMaskProvider
from .domain_randomization import DomainRandomization

__all__ = [
    "DomainRandomization",
    "PhotometricRandomize",
    "BackgroundBank",
    "MaskGuidedBackground",
    "NullMaskProvider",
    "PrecomputedMaskProvider",
]
