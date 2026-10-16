"""Candidate normalization, registration checks, and proposed-ink extraction.

Supported provider output convention
------------------------------------
A provider returns the complete page in portrait orientation with the same
framing as the source: no crop, border, rotation, or padding. Its size may
differ, but its aspect ratio must match the canonical raster within
``aspect_ratio_tolerance``. Such output is mapped to canonical pixels by one
recorded uniform scale. Anything else is rejected as unsupported geometry;
stretching a cropped or padded image would silently misregister new ink.

Crop framing is the one declared exception: when the provider output carries a
``canvas_box``, it depicts exactly that window of the page. It must match the
window's aspect ratio, is scaled into the window, and is pasted over the source
page. Registration is measured inside the window only, because the untouched
rest of the page would otherwise always look perfectly registered.

New-ink extraction (registered-image comparison)
------------------------------------------------
Raw pixel subtraction is not evidence of new art: models redraw existing
strokes slightly thicker or offset, add paper noise, and may echo guidance
colours. Extraction therefore keeps only pixels that are

1. dark and nearly neutral (black line art, not tinted guidance colours),
2. outside a tolerance band around the original ink, so redrawn originals are
   never reprinted, and
3. part of connected components larger than a minimum physical area.

Registration is checked first against the original ink: the estimated global
shift must be small and most original ink must still be present nearby. When
the page has too little ink to measure, registration is recorded as unverified.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout, mm_to_px, px_to_mm

from .framing import PixelBox, full_box
from .models import (
    BinaryMask,
    ExtractionReport,
    NormalizationRecord,
    NormalizedCandidate,
    RGBImage,
    RegistrationReport,
    readonly_copy,
    require_rgb_image,
)


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    aspect_ratio_tolerance: float = 0.01
    min_output_side_px: int = 128
    dark_ink_threshold: int = 128
    max_ink_chroma: int = 60
    source_redraw_tolerance_mm: float = 1.0
    max_registration_shift_mm: float = 0.5
    min_source_retention: float = 0.85
    min_source_ink_pixels_for_registration: int = 50
    min_component_area_mm2: float = 0.5
    registration_max_side_px: int = 1024

    def __post_init__(self) -> None:
        if not 0 <= self.aspect_ratio_tolerance < 1:
            raise ValueError("aspect_ratio_tolerance must be in [0, 1)")
        if self.min_output_side_px < 1:
            raise ValueError("min_output_side_px must be positive")
        if not 0 < self.dark_ink_threshold <= 255:
            raise ValueError("dark_ink_threshold must be in (0, 255]")
        if not 0 <= self.max_ink_chroma <= 255:
            raise ValueError("max_ink_chroma must be in [0, 255]")
        if self.source_redraw_tolerance_mm < 0 or self.max_registration_shift_mm < 0:
            raise ValueError("tolerances cannot be negative")
        if self.max_registration_shift_mm > self.source_redraw_tolerance_mm:
            # A shift larger than the redraw band would let displaced originals pass as new ink.
            raise ValueError("max_registration_shift_mm cannot exceed source_redraw_tolerance_mm")
        if not 0 <= self.min_source_retention <= 1:
            raise ValueError("min_source_retention must be in [0, 1]")
        if self.min_component_area_mm2 < 0:
            raise ValueError("min_component_area_mm2 cannot be negative")
        if self.registration_max_side_px < 64:
            raise ValueError("registration_max_side_px must be at least 64")


@dataclass(frozen=True, slots=True, eq=False)
class NormalizationOutcome:
    """Either a normalized candidate or failure reasons; the record is always present."""

    record: NormalizationRecord
    candidate: NormalizedCandidate | None

    @property
    def accepted(self) -> bool:
        return self.candidate is not None


def _identity_rows() -> tuple[tuple[float, float, float], ...]:
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rejection(
    input_shape: tuple[int, ...],
    reasons: list[str],
    *,
    transform: tuple[tuple[float, float, float], ...] | None = None,
    aspect_error: float = 0.0,
    registration: RegistrationReport | None = None,
    canvas_box: PixelBox | None = None,
) -> NormalizationOutcome:
    record = NormalizationRecord(
        input_shape=input_shape,
        raw_to_canonical=transform or _identity_rows(),
        aspect_ratio_error=aspect_error,
        registration=registration,
        extraction=None,
        failure_reasons=tuple(reasons),
        canvas_box=None if canvas_box is None else canvas_box.as_tuple(),
    )
    return NormalizationOutcome(record, None)


def to_rgb_uint8(image: np.ndarray) -> RGBImage | None:
    """Convert provider pixels to RGB; alpha is composited on white paper.

    Returns ``None`` for layouts that have no unambiguous RGB interpretation.
    """
    if image.dtype != np.uint8:
        return None
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        composited = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return np.clip(np.rint(composited), 0, 255).astype(np.uint8)
    return None


def _scale_transform(
    scale_x: float, scale_y: float, offset_x: float = 0.0, offset_y: float = 0.0
) -> tuple[tuple[float, float, float], ...]:
    return ((scale_x, 0.0, offset_x), (0.0, scale_y, offset_y), (0.0, 0.0, 1.0))


def dark_neutral_ink(image: RGBImage, config: ExtractionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks of (dark neutral ink, dark coloured pixels)."""
    red, green, blue = cv2.split(image)
    luminance = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    chroma = cv2.subtract(cv2.max(cv2.max(red, green), blue), cv2.min(cv2.min(red, green), blue))
    dark = luminance < config.dark_ink_threshold
    neutral = chroma <= config.max_ink_chroma
    return dark & neutral, dark & ~neutral


def _disk(radius_px: float) -> np.ndarray:
    size = 2 * int(np.ceil(radius_px)) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _within_distance(mask: np.ndarray, distance_px: float) -> np.ndarray:
    """Boolean pixels whose centre is within ``distance_px`` of ``mask``."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    distances = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distances <= distance_px


def _downsample_for_registration(mask: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    factor = max(mask.shape) / float(max_side)
    if factor <= 1.0:
        return mask.astype(np.float32), 1.0
    size = (int(round(mask.shape[1] / factor)), int(round(mask.shape[0] / factor)))
    small = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return small, factor


def check_registration(
    source_ink: np.ndarray,
    candidate_ink: np.ndarray,
    layout: A4CalibrationLayout,
    config: ExtractionConfig,
) -> RegistrationReport:
    """Estimate global movement of original ink and how much of it survived in place."""
    source_pixels = int(np.count_nonzero(source_ink))
    if source_pixels < config.min_source_ink_pixels_for_registration:
        return RegistrationReport(False, (0.0, 0.0), 0.0, 0.0, source_pixels, 1.0)

    source_small, factor = _downsample_for_registration(source_ink, config.registration_max_side_px)
    candidate_small, _ = _downsample_for_registration(candidate_ink, config.registration_max_side_px)
    (shift_x, shift_y), response = cv2.phaseCorrelate(source_small, candidate_small)
    shift_px = (float(shift_x) * factor, float(shift_y) * factor)
    shift_mm = px_to_mm(float(np.hypot(*shift_px)), layout.dpi)

    tolerance_px = mm_to_px(config.source_redraw_tolerance_mm, layout.dpi)
    near_candidate = _within_distance(candidate_ink, tolerance_px)
    retained = int(np.count_nonzero(source_ink & near_candidate))
    return RegistrationReport(
        verified=True,
        estimated_shift_px=shift_px,
        estimated_shift_mm=shift_mm,
        phase_response=float(response),
        source_ink_pixels=source_pixels,
        source_retention=retained / source_pixels,
    )


def _registration_failures(report: RegistrationReport, config: ExtractionConfig) -> list[str]:
    if not report.verified:
        return []
    reasons = []
    if report.estimated_shift_mm > config.max_registration_shift_mm:
        reasons.append(
            f"source content moved by {report.estimated_shift_mm:.2f} mm, above "
            f"{config.max_registration_shift_mm:.2f} mm"
        )
    if report.source_retention < config.min_source_retention:
        reasons.append(
            f"only {report.source_retention:.1%} of original ink is preserved in place, below "
            f"{config.min_source_retention:.1%}"
        )
    return reasons


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 1 or not mask.any():
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_pixels
    return keep[labels]


def _min_component_pixels(layout: A4CalibrationLayout, config: ExtractionConfig) -> int:
    pixel_area_mm2 = px_to_mm(1.0, layout.dpi) ** 2
    return int(np.ceil(config.min_component_area_mm2 / pixel_area_mm2))


def extract_new_ink(
    candidate_rgb: RGBImage,
    existing_ink_mask: BinaryMask,
    layout: A4CalibrationLayout,
    config: ExtractionConfig,
    additions_layer: np.ndarray | None = None,
    ink_classes: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[BinaryMask, ExtractionReport]:
    """Return ``candidate_new_ink_mask`` (0/255) and counts of what was discarded.

    ``ink_classes`` may pass a precomputed ``dark_neutral_ink`` result.
    """
    neutral_ink, coloured = ink_classes or dark_neutral_ink(candidate_rgb, config)
    if additions_layer is not None:
        proposed = additions_layer & ~coloured
        method = "explicit_additions_layer"
    else:
        proposed = neutral_ink
        method = "registered_image_comparison"
    candidate_pixels = int(np.count_nonzero(proposed))

    redraw_zone = _within_distance(existing_ink_mask > 0, mm_to_px(config.source_redraw_tolerance_mm, layout.dpi))
    away_from_source = proposed & ~redraw_zone
    cleaned = _remove_small_components(away_from_source, _min_component_pixels(layout, config))
    report = ExtractionReport(
        method=method,
        candidate_ink_pixels=candidate_pixels,
        discarded_source_redraw_pixels=int(np.count_nonzero(proposed & redraw_zone)),
        discarded_noise_pixels=int(np.count_nonzero(away_from_source & ~cleaned)),
        discarded_coloured_pixels=int(np.count_nonzero(coloured)),
        new_ink_pixels=int(np.count_nonzero(cleaned)),
    )
    return cleaned.astype(np.uint8) * 255, report


def normalize_and_extract(
    raw_image: np.ndarray,
    existing_ink_mask: BinaryMask,
    layout: A4CalibrationLayout,
    config: ExtractionConfig,
    additions_mask: np.ndarray | None = None,
    source_page: RGBImage | None = None,
    canvas_box: tuple[int, int, int, int] | None = None,
) -> NormalizationOutcome:
    """Map provider output to canonical pixels and extract proposed new ink.

    When available, ``source_page`` supplies the registration reference. Track
    A deliberately detects a wider range of faint ink than this module's
    dark-neutral extraction threshold. Comparing that broad mask directly with
    dark candidate pixels makes unchanged antialiased source ink look missing.
    Registration therefore compares the same dark-neutral class on both source
    and candidate, while extraction and protection continue to use Track A's
    complete existing-ink mask.

    ``canvas_box`` declares crop framing (see the module docstring). It requires
    ``source_page``, which fills the page outside the window.
    """
    input_shape = tuple(int(value) for value in raw_image.shape)
    rgb = to_rgb_uint8(raw_image)
    if rgb is None:
        return _rejection(input_shape, [f"unsupported pixel layout {input_shape} / {raw_image.dtype}"])
    height, width = rgb.shape[:2]
    if min(height, width) < config.min_output_side_px:
        return _rejection(input_shape, [f"output {width}x{height} is smaller than {config.min_output_side_px} px"])

    page_box = full_box(layout.image_shape)
    window = page_box
    if canvas_box is not None:
        if source_page is None:
            raise ValueError("crop-framed output requires source_page")
        window = PixelBox(*canvas_box)
        if not window.fits(layout.image_shape):
            return _rejection(
                input_shape, [f"canvas box {window.as_tuple()} lies outside the canonical page"], canvas_box=window
            )
    cropped = window != page_box
    target_height, target_width = window.height, window.width
    expected_aspect = target_width / target_height
    aspect_error = abs((width / height) / expected_aspect - 1.0)
    if aspect_error > config.aspect_ratio_tolerance:
        expected = f"the declared {target_width}x{target_height} window" if cropped else "A4"
        unsupported = "" if cropped else "; cropped, padded, or rotated output is unsupported"
        return _rejection(
            input_shape,
            [
                f"aspect ratio {width / height:.4f} differs from {expected} {expected_aspect:.4f} by "
                f"{aspect_error:.2%}{unsupported}"
            ],
            aspect_error=aspect_error,
            canvas_box=window if cropped else None,
        )

    transform = _scale_transform(target_width / width, target_height / height, float(window.x), float(window.y))
    if (height, width) == (target_height, target_width):
        window_rgb = rgb
    else:
        window_rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if cropped:
        assert source_page is not None
        require_rgb_image(source_page, layout.image_shape, "source_page")
        canonical = source_page.copy()
        canonical[window.slices()] = window_rgb
    else:
        canonical = window_rgb

    existing = existing_ink_mask > 0
    window_neutral, window_coloured = dark_neutral_ink(window_rgb, config)
    registration_source = existing
    if source_page is not None:
        require_rgb_image(source_page, layout.image_shape, "source_page")
        registration_source = existing & dark_neutral_ink(source_page, config)[0]
    registration = check_registration(window.crop(registration_source), window_neutral, layout, config)
    failures = _registration_failures(registration, config)
    record_box = window.as_tuple() if cropped else None
    if failures:
        return _rejection(
            input_shape,
            failures,
            transform=transform,
            aspect_error=aspect_error,
            registration=registration,
            canvas_box=window if cropped else None,
        )

    # Extraction runs on the full page so the redraw band also covers original
    # ink just outside the window; proposals can only come from the window.
    neutral_ink = np.zeros(layout.image_shape, dtype=bool)
    coloured = np.zeros(layout.image_shape, dtype=bool)
    neutral_ink[window.slices()] = window_neutral
    coloured[window.slices()] = window_coloured
    additions_layer = None
    if additions_mask is not None:
        resized = cv2.resize(
            (np.asarray(additions_mask) > 0).astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        additions_layer = np.zeros(layout.image_shape, dtype=bool)
        additions_layer[window.slices()] = resized > 0
    new_ink, extraction = extract_new_ink(
        canonical, existing_ink_mask, layout, config, additions_layer, (neutral_ink, coloured)
    )
    record = NormalizationRecord(
        input_shape=input_shape,
        raw_to_canonical=transform,
        aspect_ratio_error=aspect_error,
        registration=registration,
        extraction=extraction,
        canvas_box=record_box,
    )
    return NormalizationOutcome(record, NormalizedCandidate(readonly_copy(canonical), readonly_copy(new_ink)))
