"""Shared data models for deterministic PrintGPT masking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray


RGBArray: TypeAlias = NDArray[np.uint8]
BinaryMask: TypeAlias = NDArray[np.uint8]
JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class MaskingConfig:
    """Configuration for ink detection and printable-region construction."""

    absolute_ink_threshold: int = 220
    local_contrast_threshold: int = 12
    background_blur_mm: float = 2.0
    median_blur_px: int = 0
    min_existing_component_pixels: int = 0
    clearance_mm: float = 0.5
    zero_clearance_test_mode: bool = False
    printer_margin_mm: float = 10.0
    layout_guide_clearance_mm: float = 0.5
    exclude_calibration_labels: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.absolute_ink_threshold <= 255:
            raise ValueError("absolute_ink_threshold must be in [0, 255]")
        if not 0 <= self.local_contrast_threshold <= 255:
            raise ValueError("local_contrast_threshold must be in [0, 255]")
        if self.background_blur_mm <= 0:
            raise ValueError("background_blur_mm must be positive")
        if self.median_blur_px < 0 or (self.median_blur_px not in (0, 1) and self.median_blur_px % 2 == 0):
            raise ValueError("median_blur_px must be zero, one, or a positive odd integer")
        if self.min_existing_component_pixels < 0:
            raise ValueError("min_existing_component_pixels cannot be negative")
        if self.clearance_mm < 0:
            raise ValueError("clearance_mm cannot be negative")
        if self.clearance_mm == 0 and not self.zero_clearance_test_mode:
            raise ValueError("zero clearance requires zero_clearance_test_mode=True")
        if self.printer_margin_mm < 0:
            raise ValueError("printer_margin_mm cannot be negative")
        if self.layout_guide_clearance_mm < 0:
            raise ValueError("layout_guide_clearance_mm cannot be negative")

    @property
    def effective_clearance_mm(self) -> float:
        """Return zero only when the explicit test mode is enabled."""
        return 0.0 if self.zero_clearance_test_mode else self.clearance_mm

    def to_dict(self) -> dict[str, JSONValue]:
        result = asdict(self)
        result["effective_clearance_mm"] = self.effective_clearance_mm
        return result


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Thresholds used to accept or reject a proposed addition."""

    min_final_ink_pixels: int = 25
    max_final_coverage_fraction: float = 0.20
    max_removed_fraction: float = 0.25
    small_component_max_pixels: int = 20
    max_small_component_fraction: float = 0.50
    max_connected_components: int | None = None
    reject_any_forbidden_intersection: bool = False

    def __post_init__(self) -> None:
        if self.min_final_ink_pixels < 0:
            raise ValueError("min_final_ink_pixels cannot be negative")
        for name in (
            "max_final_coverage_fraction",
            "max_removed_fraction",
            "max_small_component_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.small_component_max_pixels < 0:
            raise ValueError("small_component_max_pixels cannot be negative")
        if self.max_connected_components is not None and self.max_connected_components < 0:
            raise ValueError("max_connected_components cannot be negative")

    def to_dict(self) -> dict[str, JSONValue]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MaskBundle:
    """Canonical masks and metadata produced before candidate validation."""

    existing_ink_mask: BinaryMask
    protected_mask: BinaryMask
    allowed_region_mask: BinaryMask
    safe_page_interior_mask: BinaryMask
    layout_exclusion_mask: BinaryMask
    printable_region_mask: BinaryMask
    geometry: dict[str, JSONValue]
    config: dict[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Deterministic validation decision and clipped printable addition."""

    accepted: bool
    reasons: list[str]
    metrics: dict[str, JSONValue]
    sanitized_new_ink_mask: BinaryMask

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe decision data without expanding the raster mask."""
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "sanitized_new_ink_mask": {
                "shape": list(self.sanitized_new_ink_mask.shape),
                "dtype": str(self.sanitized_new_ink_mask.dtype),
                "ink_pixels": int(np.count_nonzero(self.sanitized_new_ink_mask)),
            },
        }
