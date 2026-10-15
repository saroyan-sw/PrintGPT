"""Diagnostic visualizations for PrintGPT mask decisions."""

from __future__ import annotations

import numpy as np

from printgpt_contracts import BinaryMask, MaskBundle, RGBArray, ValidationResult

from .core import _validate_binary_mask, _validate_rgb_image, compose_preview


def _color_mask(mask: BinaryMask, color: tuple[int, int, int]) -> RGBArray:
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask == 255] = color
    return image


def compose_diagnostic_overlay(
    canonical_page: RGBArray,
    candidate_new_ink_mask: BinaryMask,
    masks: MaskBundle,
    validation: ValidationResult,
) -> RGBArray:
    """Tint protection orange, removed proposals red, and accepted ink black."""
    shape = masks.printable_region_mask.shape
    _validate_rgb_image(canonical_page, shape, "canonical_page")
    _validate_binary_mask(candidate_new_ink_mask, shape, "candidate_new_ink_mask")
    _validate_binary_mask(validation.sanitized_new_ink_mask, shape, "sanitized_new_ink_mask")
    overlay = canonical_page.copy()
    protected = masks.protected_mask == 255
    removed = (candidate_new_ink_mask == 255) & (validation.sanitized_new_ink_mask == 0)
    overlay[protected] = (
        0.55 * overlay[protected].astype(np.float32) + 0.45 * np.asarray((255, 140, 0))
    ).astype(np.uint8)
    overlay[removed] = (220, 30, 30)
    overlay[validation.sanitized_new_ink_mask == 255] = (0, 0, 0)
    return overlay


def diagnostic_images(
    canonical_page: RGBArray,
    candidate_new_ink_mask: BinaryMask,
    masks: MaskBundle,
    validation: ValidationResult,
) -> dict[str, RGBArray]:
    """Return named RGB images that explain protection and clipping."""
    shape = masks.printable_region_mask.shape
    _validate_binary_mask(candidate_new_ink_mask, shape, "candidate_new_ink_mask")
    _validate_binary_mask(validation.sanitized_new_ink_mask, shape, "sanitized_new_ink_mask")
    removed = ((candidate_new_ink_mask > 0) & (validation.sanitized_new_ink_mask == 0)).astype(np.uint8) * 255
    return {
        "existing_ink": _color_mask(masks.existing_ink_mask, (0, 0, 0)),
        "protected_ink": _color_mask(masks.protected_mask, (255, 140, 0)),
        "allowed_region": _color_mask(masks.allowed_region_mask, (50, 150, 255)),
        "printable_region": _color_mask(masks.printable_region_mask, (50, 180, 80)),
        "removed_candidate_pixels": _color_mask(removed, (220, 30, 30)),
        "sanitized_new_ink": _color_mask(validation.sanitized_new_ink_mask, (0, 0, 0)),
        "final_preview": compose_preview(canonical_page, validation.sanitized_new_ink_mask),
        "decision_overlay": compose_diagnostic_overlay(
            canonical_page, candidate_new_ink_mask, masks, validation
        ),
    }
