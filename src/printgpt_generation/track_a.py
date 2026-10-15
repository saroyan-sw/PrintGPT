"""Boundary with Track A's shared masking contract.

Track B consumes ``MaskBundle`` and ``ValidationResult`` structurally through
the protocols below, so it never defines a competing shared-model package.
``TrackAValidator`` delegates to ``printgpt_masking`` once that package is
installed. The ``Simulated*`` doubles exist only so this track runs before
integration; their output is labelled simulated and is not print-ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_px

from .models import (
    BinaryMask,
    JSONValue,
    RGBImage,
    ValidationSummary,
    readonly_copy,
    require_binary_mask,
    require_rgb_image,
)


class MaskBundleLike(Protocol):
    """Fields of Track A's ``MaskBundle`` that Track B reads."""

    @property
    def existing_ink_mask(self) -> BinaryMask: ...
    @property
    def protected_mask(self) -> BinaryMask: ...
    @property
    def allowed_region_mask(self) -> BinaryMask: ...
    @property
    def printable_region_mask(self) -> BinaryMask: ...
    @property
    def geometry(self) -> dict[str, JSONValue]: ...
    @property
    def config(self) -> dict[str, JSONValue]: ...


class ValidationResultLike(Protocol):
    @property
    def accepted(self) -> bool: ...
    @property
    def reasons(self) -> list[str]: ...
    @property
    def metrics(self) -> dict[str, JSONValue]: ...
    @property
    def sanitized_new_ink_mask(self) -> BinaryMask: ...


class CandidateValidator(Protocol):
    """Deterministic enforcement and preview composition owned by Track A."""

    @property
    def simulated(self) -> bool: ...

    def validate(self, candidate_new_ink_mask: BinaryMask, masks: MaskBundleLike) -> ValidationResultLike: ...

    def compose_preview(self, canonical_page: RGBImage, sanitized_new_ink_mask: BinaryMask) -> RGBImage: ...


def require_mask_bundle(masks: MaskBundleLike, expected_shape: tuple[int, int]) -> None:
    """Raise ``ValueError`` unless every mask Track B reads follows the contract."""
    for name in ("existing_ink_mask", "protected_mask", "allowed_region_mask", "printable_region_mask"):
        require_binary_mask(getattr(masks, name, None), expected_shape, f"masks.{name}")
    if not np.any(masks.allowed_region_mask):
        # Missing allowed-region information is an error, not permission to use the page.
        raise ValueError("masks.allowed_region_mask is empty")


def summarize_validation(result: ValidationResultLike, *, simulated: bool) -> ValidationSummary:
    return ValidationSummary(
        accepted=bool(result.accepted),
        reasons=tuple(str(reason) for reason in result.reasons),
        metrics=dict(result.metrics),
        simulated=simulated,
    )


@dataclass(frozen=True, slots=True, eq=False)
class SimulatedMaskBundle:
    """Test double with ``MaskBundle``'s field names. Not print-ready."""

    existing_ink_mask: BinaryMask
    protected_mask: BinaryMask
    allowed_region_mask: BinaryMask
    safe_page_interior_mask: BinaryMask
    layout_exclusion_mask: BinaryMask
    printable_region_mask: BinaryMask
    geometry: dict[str, JSONValue]
    config: dict[str, JSONValue] = field(default_factory=dict)


def build_simulated_masks(
    canonical_page: RGBImage,
    allowed_region_mask: BinaryMask,
    *,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    clearance_mm: float = 0.5,
    margin_mm: float = 10.0,
    ink_threshold: int = 200,
) -> SimulatedMaskBundle:
    """Approximate Track A's masks with a plain threshold; for offline tests only."""
    require_rgb_image(canonical_page, layout.image_shape, "canonical_page")
    require_binary_mask(allowed_region_mask, layout.image_shape, "allowed_region_mask")
    existing = cv2.cvtColor(canonical_page, cv2.COLOR_RGB2GRAY) < ink_threshold
    if existing.any():
        distances = cv2.distanceTransform((~existing).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        protected = distances <= mm_to_px(clearance_mm, layout.dpi)
    else:
        protected = existing
    margin = int(round(mm_to_px(margin_mm, layout.dpi)))
    interior = np.zeros(layout.image_shape, dtype=bool)
    interior[margin : layout.image_shape[0] - margin, margin : layout.image_shape[1] - margin] = True
    printable = (allowed_region_mask > 0) & interior & ~protected
    exclusions = np.zeros(layout.image_shape, dtype=bool)

    def as_mask(values: np.ndarray) -> BinaryMask:
        return readonly_copy(values.astype(np.uint8) * 255)

    return SimulatedMaskBundle(
        existing_ink_mask=as_mask(existing),
        protected_mask=as_mask(protected),
        allowed_region_mask=readonly_copy(allowed_region_mask),
        safe_page_interior_mask=as_mask(interior),
        layout_exclusion_mask=as_mask(exclusions),
        printable_region_mask=as_mask(printable),
        geometry={
            "layout_id": layout.layout_id,
            "dpi": float(layout.dpi),
            "image_shape_rows_columns": [int(layout.image_shape[0]), int(layout.image_shape[1])],
            "mask_membership_value": 255,
            "simulated": True,
        },
        config={"clearance_mm": clearance_mm, "printer_margin_mm": margin_mm, "simulated": True},
    )


@dataclass(frozen=True, slots=True, eq=False)
class SimulatedValidationResult:
    accepted: bool
    reasons: list[str]
    metrics: dict[str, JSONValue]
    sanitized_new_ink_mask: BinaryMask


def compose_black_preview(canonical_page: RGBImage, sanitized_new_ink_mask: BinaryMask) -> RGBImage:
    """Source page with black ink where the sanitized mask is 255."""
    require_rgb_image(canonical_page, canonical_page.shape[:2], "canonical_page")
    require_binary_mask(sanitized_new_ink_mask, canonical_page.shape[:2], "sanitized_new_ink_mask")
    preview = canonical_page.copy()
    preview[sanitized_new_ink_mask == 255] = (0, 0, 0)
    return readonly_copy(preview)


@dataclass(frozen=True, slots=True)
class SimulatedValidator:
    """Clip-and-count stand-in for ``printgpt_masking.validate_candidate``."""

    min_final_ink_pixels: int = 25
    max_removed_fraction: float = 0.25

    @property
    def simulated(self) -> bool:
        return True

    def validate(self, candidate_new_ink_mask: BinaryMask, masks: MaskBundleLike) -> SimulatedValidationResult:
        require_binary_mask(candidate_new_ink_mask, masks.printable_region_mask.shape, "candidate_new_ink_mask")
        proposed = candidate_new_ink_mask > 0
        sanitized = proposed & (masks.printable_region_mask > 0)
        proposed_count = int(np.count_nonzero(proposed))
        final_count = int(np.count_nonzero(sanitized))
        removed_fraction = (proposed_count - final_count) / proposed_count if proposed_count else 0.0
        reasons = []
        if final_count < self.min_final_ink_pixels:
            reasons.append(
                f"sanitized addition has {final_count} ink pixels, below minimum {self.min_final_ink_pixels}"
            )
        if removed_fraction > self.max_removed_fraction:
            reasons.append(
                f"removed fraction {removed_fraction:.6f} exceeds maximum {self.max_removed_fraction:.6f}"
            )
        return SimulatedValidationResult(
            accepted=not reasons,
            reasons=reasons,
            metrics={
                "simulated": True,
                "proposed_ink_count": proposed_count,
                "final_ink_count": final_count,
                "removed_pixel_count": proposed_count - final_count,
                "removed_fraction": removed_fraction,
            },
            sanitized_new_ink_mask=readonly_copy(sanitized.astype(np.uint8) * 255),
        )

    def compose_preview(self, canonical_page: RGBImage, sanitized_new_ink_mask: BinaryMask) -> RGBImage:
        return compose_black_preview(canonical_page, sanitized_new_ink_mask)


class TrackAValidator:
    """Adapter over ``printgpt_masking``; importing it is deferred until construction."""

    def __init__(self, validation_config: object | None = None) -> None:
        # Imported here so this module works before Track A is merged.
        from printgpt_contracts import ValidationConfig
        from printgpt_masking import compose_preview, validate_candidate

        if validation_config is not None and not isinstance(validation_config, ValidationConfig):
            raise ValueError("validation_config must be a printgpt_contracts.ValidationConfig")
        self._config = validation_config or ValidationConfig()
        self._validate = validate_candidate
        self._compose = compose_preview

    @property
    def simulated(self) -> bool:
        return False

    def validate(self, candidate_new_ink_mask: BinaryMask, masks: MaskBundleLike) -> ValidationResultLike:
        return self._validate(candidate_new_ink_mask, masks, config=self._config)

    def compose_preview(self, canonical_page: RGBImage, sanitized_new_ink_mask: BinaryMask) -> RGBImage:
        return self._compose(canonical_page, sanitized_new_ink_mask)
