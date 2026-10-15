"""Deterministic masks and validation for safe PrintGPT additions."""

from printgpt_contracts import MaskBundle, MaskingConfig, ValidationConfig, ValidationResult

from .core import build_masks, compose_preview, detect_existing_ink, validate_candidate
from .diagnostics import compose_diagnostic_overlay, diagnostic_images

__all__ = [
    "MaskBundle",
    "MaskingConfig",
    "ValidationConfig",
    "ValidationResult",
    "build_masks",
    "compose_preview",
    "compose_diagnostic_overlay",
    "detect_existing_ink",
    "diagnostic_images",
    "validate_candidate",
]
