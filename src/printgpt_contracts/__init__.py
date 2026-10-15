"""Shared contracts between PrintGPT masking and generation tracks."""

from .models import (
    BinaryMask,
    JSONValue,
    MaskBundle,
    MaskingConfig,
    RGBArray,
    ValidationConfig,
    ValidationResult,
)

__all__ = [
    "BinaryMask",
    "JSONValue",
    "MaskBundle",
    "MaskingConfig",
    "RGBArray",
    "ValidationConfig",
    "ValidationResult",
]
