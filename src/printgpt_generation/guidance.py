"""Region-guidance images shown to the generator and critic.

Guidance uses light tints and a saturated outline. Both are rejected by
extraction's dark-and-neutral ink test, so a model that copies guidance marks
into its output cannot turn them into printable content.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from .models import BinaryMask, RGBImage, readonly_copy


PRINTABLE_TINT_RGB: Final[tuple[int, int, int]] = (150, 230, 150)
FORBIDDEN_TINT_RGB: Final[tuple[int, int, int]] = (240, 150, 150)
OUTLINE_RGB: Final[tuple[int, int, int]] = (255, 0, 255)
_TINT_WEIGHT: Final[float] = 0.35


def _blend(image: np.ndarray, region: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    result = image.astype(np.float32)
    tint = np.asarray(colour, dtype=np.float32)
    result[region] = result[region] * (1.0 - _TINT_WEIGHT) + tint * _TINT_WEIGHT
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def render_region_guidance(
    canonical_page: RGBImage,
    printable_region_mask: BinaryMask,
) -> RGBImage:
    """Tint printable paper green, all other paper red, and outline the printable region."""
    printable = printable_region_mask > 0
    # Leave existing ink untinted so the model still sees the original strokes clearly.
    paper = cv2.cvtColor(canonical_page, cv2.COLOR_RGB2GRAY) >= 128
    guidance = _blend(canonical_page, printable & paper, PRINTABLE_TINT_RGB)
    guidance = _blend(guidance, ~printable & paper, FORBIDDEN_TINT_RGB)
    contours, _ = cv2.findContours(printable.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(1, round(max(guidance.shape[:2]) / 1200))
    cv2.drawContours(guidance, contours, -1, OUTLINE_RGB, thickness)
    return readonly_copy(guidance)
