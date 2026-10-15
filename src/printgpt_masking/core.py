"""Deterministic existing-content masking and candidate validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor

import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_px, px_to_mm
from printgpt_contracts import (
    BinaryMask,
    JSONValue,
    MaskBundle,
    MaskingConfig,
    RGBArray,
    ValidationConfig,
    ValidationResult,
)


@dataclass(frozen=True, slots=True)
class _ComponentMetrics:
    count: int
    small_pixel_count: int
    small_fraction: float


def _validate_rgb_image(image: np.ndarray, expected_shape: tuple[int, int], name: str) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    required_shape = (*expected_shape, 3)
    if image.shape != required_shape:
        raise ValueError(f"{name} must have shape {required_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {image.dtype}")


def _validate_binary_mask(mask: np.ndarray | None, expected_shape: tuple[int, int], name: str) -> None:
    if mask is None:
        raise ValueError(f"{name} is required")
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
    if mask.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {mask.dtype}")
    if not bool(np.all((mask == 0) | (mask == 255))):
        raise ValueError(f"{name} values must be exactly 0 or 255")


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array.copy())
    result.flags.writeable = False
    return result


def _odd_kernel_size(value_px: float) -> int:
    rounded = max(3, int(round(value_px)))
    return rounded if rounded % 2 == 1 else rounded + 1


def _remove_small_components(mask: BinaryMask, minimum_pixels: int) -> BinaryMask:
    if minimum_pixels <= 1 or not np.any(mask):
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= minimum_pixels).astype(np.uint8) * 255
    return keep[labels]


def detect_existing_ink(
    canonical_page: RGBArray,
    *,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    config: MaskingConfig = MaskingConfig(),
) -> BinaryMask:
    """Detect dark line art against global and locally varying light paper."""
    _validate_rgb_image(canonical_page, layout.image_shape, "canonical_page")
    gray = cv2.cvtColor(canonical_page, cv2.COLOR_RGB2GRAY)
    if config.median_blur_px > 1:
        detection_gray = cv2.medianBlur(gray, config.median_blur_px)
    else:
        detection_gray = gray
    kernel = _odd_kernel_size(mm_to_px(config.background_blur_mm, layout.dpi))
    local_background = cv2.GaussianBlur(detection_gray, (kernel, kernel), 0)
    local_darkness = local_background.astype(np.int16) - detection_gray.astype(np.int16)
    detected = (detection_gray <= config.absolute_ink_threshold) | (
        local_darkness >= config.local_contrast_threshold
    )
    mask = detected.astype(np.uint8) * 255
    return _remove_small_components(mask, config.min_existing_component_pixels)


def _protect_existing_ink(
    existing_ink_mask: BinaryMask,
    clearance_mm: float,
    layout: A4CalibrationLayout,
) -> BinaryMask:
    if clearance_mm == 0:
        return existing_ink_mask.copy()
    clearance_px = mm_to_px(clearance_mm, layout.dpi)
    background = (existing_ink_mask == 0).astype(np.uint8)
    distances = cv2.distanceTransform(background, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return (distances <= clearance_px).astype(np.uint8) * 255


def _safe_page_interior(layout: A4CalibrationLayout, margin_mm: float) -> BinaryMask:
    if 2.0 * margin_mm >= min(layout.page_width_mm, layout.page_height_mm):
        raise ValueError("printer_margin_mm leaves no page interior")
    x0 = ceil(mm_to_px(margin_mm, layout.dpi))
    y0 = ceil(mm_to_px(margin_mm, layout.dpi))
    x1 = floor(mm_to_px(layout.page_width_mm - margin_mm, layout.dpi))
    y1 = floor(mm_to_px(layout.page_height_mm - margin_mm, layout.dpi))
    result = np.zeros(layout.image_shape, dtype=np.uint8)
    result[y0:y1, x0:x1] = 255
    return result


def _fill_rect_mm(
    mask: BinaryMask,
    layout: A4CalibrationLayout,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    x0 = max(0, floor(mm_to_px(x, layout.dpi)))
    y0 = max(0, floor(mm_to_px(y, layout.dpi)))
    x1 = min(layout.raster_width_px, ceil(mm_to_px(x + width, layout.dpi)))
    y1 = min(layout.raster_height_px, ceil(mm_to_px(y + height, layout.dpi)))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255


def _layout_exclusions(layout: A4CalibrationLayout, config: MaskingConfig) -> BinaryMask:
    result = np.zeros(layout.image_shape, dtype=np.uint8)
    quiet = layout.marker_quiet_zone_mm
    for marker in layout.markers:
        bounds = marker.bounds_mm
        _fill_rect_mm(
            result,
            layout,
            bounds.x - quiet,
            bounds.y - quiet,
            bounds.width + 2.0 * quiet,
            bounds.height + 2.0 * quiet,
        )

    guide = config.layout_guide_clearance_mm
    if guide > 0:
        margin = layout.protected_margin_mm
        _fill_rect_mm(result, layout, margin - guide, margin - guide, layout.page_width_mm - 2 * margin + 2 * guide, 2 * guide)
        _fill_rect_mm(result, layout, margin - guide, layout.page_height_mm - margin - guide, layout.page_width_mm - 2 * margin + 2 * guide, 2 * guide)
        _fill_rect_mm(result, layout, margin - guide, margin - guide, 2 * guide, layout.page_height_mm - 2 * margin + 2 * guide)
        _fill_rect_mm(result, layout, layout.page_width_mm - margin - guide, margin - guide, 2 * guide, layout.page_height_mm - 2 * margin + 2 * guide)

    if config.exclude_calibration_labels:
        # These regions match the labels rendered by calibration_page.py.
        _fill_rect_mm(result, layout, 55.0, 11.0, 100.0, 14.0)
        _fill_rect_mm(result, layout, 20.0, 274.0, 170.0, 14.0)
        for marker in layout.markers:
            center_x = marker.center_mm.x
            if marker.role.startswith("top"):
                _fill_rect_mm(result, layout, center_x - 9.0, marker.bounds_mm.bottom, 18.0, 7.0)
            else:
                _fill_rect_mm(result, layout, center_x - 9.0, marker.bounds_mm.y - 9.0, 18.0, 7.0)
    return result


def build_masks(
    canonical_page: RGBArray,
    allowed_region_mask: BinaryMask,
    *,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    config: MaskingConfig = MaskingConfig(),
) -> MaskBundle:
    """Build existing, protected, exclusion, and final printable masks."""
    _validate_rgb_image(canonical_page, layout.image_shape, "canonical_page")
    _validate_binary_mask(allowed_region_mask, layout.image_shape, "allowed_region_mask")
    existing = detect_existing_ink(canonical_page, layout=layout, config=config)
    protected = _protect_existing_ink(existing, config.effective_clearance_mm, layout)
    allowed = allowed_region_mask.copy()
    safe_interior = _safe_page_interior(layout, config.printer_margin_mm)
    layout_exclusion = _layout_exclusions(layout, config)
    forbidden = (protected > 0) | (safe_interior == 0) | (layout_exclusion > 0)
    printable = ((allowed > 0) & ~forbidden).astype(np.uint8) * 255
    geometry: dict[str, JSONValue] = {
        "layout_id": layout.layout_id,
        "dpi": float(layout.dpi),
        "page_width_mm": float(layout.page_width_mm),
        "page_height_mm": float(layout.page_height_mm),
        "image_shape_rows_columns": [int(layout.image_shape[0]), int(layout.image_shape[1])],
        "mask_membership_value": 255,
        "origin": "top-left",
        "array_order": "row-column",
    }
    return MaskBundle(
        existing_ink_mask=_readonly_copy(existing),
        protected_mask=_readonly_copy(protected),
        allowed_region_mask=_readonly_copy(allowed),
        safe_page_interior_mask=_readonly_copy(safe_interior),
        layout_exclusion_mask=_readonly_copy(layout_exclusion),
        printable_region_mask=_readonly_copy(printable),
        geometry=geometry,
        config=config.to_dict(),
    )


def _validate_mask_bundle(masks: MaskBundle) -> tuple[int, int]:
    shape_value = masks.geometry.get("image_shape_rows_columns")
    if not isinstance(shape_value, list) or len(shape_value) != 2:
        raise ValueError("masks geometry is missing image_shape_rows_columns")
    expected_shape = (int(shape_value[0]), int(shape_value[1]))
    for name in (
        "existing_ink_mask",
        "protected_mask",
        "allowed_region_mask",
        "safe_page_interior_mask",
        "layout_exclusion_mask",
        "printable_region_mask",
    ):
        _validate_binary_mask(getattr(masks, name), expected_shape, f"masks.{name}")
    expected_printable = (
        (masks.allowed_region_mask > 0)
        & (masks.safe_page_interior_mask > 0)
        & (masks.protected_mask == 0)
        & (masks.layout_exclusion_mask == 0)
    )
    if not np.array_equal(masks.printable_region_mask > 0, expected_printable):
        raise ValueError("masks.printable_region_mask is inconsistent with its source masks")
    return expected_shape


def _count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _component_metrics(mask: BinaryMask, small_component_max_pixels: int) -> _ComponentMetrics:
    ink_pixels = _count(mask)
    if ink_pixels == 0:
        return _ComponentMetrics(0, 0, 0.0)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    small_pixel_count = int(areas[areas <= small_component_max_pixels].sum())
    return _ComponentMetrics(component_count - 1, small_pixel_count, _fraction(small_pixel_count, ink_pixels))


def _area_mm2(pixel_count: int, dpi: float) -> float:
    pixel_side_mm = px_to_mm(1.0, dpi)
    return float(pixel_count * pixel_side_mm * pixel_side_mm)


def validate_candidate(
    candidate_new_ink_mask: BinaryMask,
    masks: MaskBundle,
    *,
    config: ValidationConfig = ValidationConfig(),
) -> ValidationResult:
    """Measure proposal violations, clip them, and validate the sanitized mask."""
    expected_shape = _validate_mask_bundle(masks)
    _validate_binary_mask(candidate_new_ink_mask, expected_shape, "candidate_new_ink_mask")
    proposed = candidate_new_ink_mask > 0
    printable = masks.printable_region_mask > 0
    sanitized = (proposed & printable).astype(np.uint8) * 255

    page_pixels = int(expected_shape[0] * expected_shape[1])
    proposed_count = _count(proposed)
    final_count = _count(sanitized)
    printable_count = _count(printable)
    allowed_count = _count(masks.allowed_region_mask)
    safe_count = _count(masks.safe_page_interior_mask)
    protected_count = _count(masks.protected_mask)
    layout_count = _count(masks.layout_exclusion_mask)
    protected_overlap = _count(proposed & (masks.protected_mask > 0))
    outside_allowed = _count(proposed & (masks.allowed_region_mask == 0))
    margin_violations = _count(proposed & (masks.safe_page_interior_mask == 0))
    layout_violations = _count(proposed & (masks.layout_exclusion_mask > 0))
    removed_count = proposed_count - final_count
    removed_fraction = _fraction(removed_count, proposed_count)
    proposed_coverage = _fraction(proposed_count, page_pixels)
    final_coverage = _fraction(final_count, printable_count)
    components = _component_metrics(sanitized, config.small_component_max_pixels)
    final_forbidden_overlap = _count((sanitized > 0) & ~printable)
    if final_forbidden_overlap:
        raise AssertionError("sanitized mask intersects a forbidden region")

    dpi_value = masks.geometry.get("dpi")
    if not isinstance(dpi_value, (int, float)) or float(dpi_value) <= 0:
        raise ValueError("masks geometry contains an invalid dpi")
    dpi = float(dpi_value)
    metrics: dict[str, JSONValue] = {
        "proposed_ink_count": proposed_count,
        "final_ink_count": final_count,
        "protected_content_overlap_pixels": protected_overlap,
        "outside_allowed_region_pixels": outside_allowed,
        "margin_violation_pixels": margin_violations,
        "layout_violation_pixels": layout_violations,
        "removed_pixel_count": removed_count,
        "removed_fraction": removed_fraction,
        "proposed_coverage": proposed_coverage,
        "final_coverage": final_coverage,
        "connected_component_count": components.count,
        "small_component_pixel_count": components.small_pixel_count,
        "small_component_fraction": components.small_fraction,
        "final_forbidden_overlap_pixels": final_forbidden_overlap,
        "fraction_denominators": {
            "removed_fraction": "proposed_ink_count; defined as 0.0 when proposed_ink_count is zero",
            "proposed_coverage": "page_pixel_count",
            "final_coverage": "printable_region_pixel_count; defined as 0.0 when printable_region_pixel_count is zero",
            "small_component_fraction": "final_ink_count; defined as 0.0 when final_ink_count is zero",
        },
        "pixel_areas": {
            "page": page_pixels,
            "allowed_region": allowed_count,
            "safe_page_interior": safe_count,
            "protected_content": protected_count,
            "layout_exclusions": layout_count,
            "printable_region": printable_count,
        },
        "areas_mm2": {
            "page": _area_mm2(page_pixels, dpi),
            "allowed_region": _area_mm2(allowed_count, dpi),
            "safe_page_interior": _area_mm2(safe_count, dpi),
            "protected_content": _area_mm2(protected_count, dpi),
            "layout_exclusions": _area_mm2(layout_count, dpi),
            "printable_region": _area_mm2(printable_count, dpi),
            "proposed_ink": _area_mm2(proposed_count, dpi),
            "final_ink": _area_mm2(final_count, dpi),
            "removed_ink": _area_mm2(removed_count, dpi),
            "protected_content_overlap": _area_mm2(protected_overlap, dpi),
            "outside_allowed_region": _area_mm2(outside_allowed, dpi),
            "margin_violations": _area_mm2(margin_violations, dpi),
            "layout_violations": _area_mm2(layout_violations, dpi),
            "small_component_pixels": _area_mm2(components.small_pixel_count, dpi),
        },
        "masking_configuration": dict(masks.config),
        "validation_thresholds": config.to_dict(),
    }

    reasons: list[str] = []
    if printable_count == 0:
        reasons.append("printable region has zero area")
    if final_count < config.min_final_ink_pixels:
        reasons.append(
            f"sanitized addition has {final_count} ink pixels, below minimum {config.min_final_ink_pixels}"
        )
    if final_coverage > config.max_final_coverage_fraction:
        reasons.append(
            f"final coverage {final_coverage:.6f} exceeds maximum {config.max_final_coverage_fraction:.6f}"
        )
    if removed_fraction > config.max_removed_fraction:
        reasons.append(
            f"removed fraction {removed_fraction:.6f} exceeds maximum {config.max_removed_fraction:.6f}"
        )
    if components.small_fraction > config.max_small_component_fraction:
        reasons.append(
            f"small-component fraction {components.small_fraction:.6f} exceeds maximum "
            f"{config.max_small_component_fraction:.6f}"
        )
    if config.max_connected_components is not None and components.count > config.max_connected_components:
        reasons.append(
            f"connected-component count {components.count} exceeds maximum {config.max_connected_components}"
        )
    if config.reject_any_forbidden_intersection and removed_count > 0:
        reasons.append(f"proposal intersects forbidden areas at {removed_count} unique pixels")
    return ValidationResult(
        accepted=not reasons,
        reasons=reasons,
        metrics=metrics,
        sanitized_new_ink_mask=_readonly_copy(sanitized),
    )


def compose_preview(canonical_page: RGBArray, sanitized_new_ink_mask: BinaryMask) -> RGBArray:
    """Return an RGB preview with sanitized additions rendered as black ink."""
    if not isinstance(canonical_page, np.ndarray) or canonical_page.ndim != 3:
        raise ValueError("canonical_page must be an RGB NumPy array")
    _validate_rgb_image(canonical_page, canonical_page.shape[:2], "canonical_page")
    _validate_binary_mask(sanitized_new_ink_mask, canonical_page.shape[:2], "sanitized_new_ink_mask")
    preview = canonical_page.copy()
    preview[sanitized_new_ink_mask == 255] = (0, 0, 0)
    return preview
