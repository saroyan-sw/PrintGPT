"""Explicit document-input modes and deterministic A4 scanner normalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from printgpt_alignment import (
    A4CalibrationLayout,
    DEFAULT_LAYOUT,
    LoadedDocumentPage,
    detect_and_rectify,
    load_document_page_info,
)


class InputMode(StrEnum):
    SCANNER_A4 = "scanner_a4"
    MARKED_PAGE = "marked_page"


class OrientationChoice(StrEnum):
    AUTO = "auto"
    ROTATE_LEFT_90 = "rotate_left_90"
    ROTATE_RIGHT_90 = "rotate_right_90"
    ROTATE_180 = "rotate_180"
    NO_ROTATION = "no_rotation"


@dataclass(frozen=True, slots=True)
class ScannerA4Tolerance:
    """Recognition tolerances for a nominal full-page A4 scanner export."""

    physical_mm: float = 3.0
    aspect_ratio_relative: float = 0.02
    raster_dimension_px: int = 12

    def __post_init__(self) -> None:
        if self.physical_mm < 0 or self.aspect_ratio_relative < 0 or self.raster_dimension_px < 0:
            raise ValueError("A4 tolerances must be non-negative")


@dataclass(frozen=True, slots=True)
class AppliedTransform:
    operation: str
    matrix: tuple[tuple[float, float, float], ...]
    source_dimensions: tuple[int, int]
    rotated_dimensions: tuple[int, int]
    resized_dimensions: tuple[int, int]
    destination_dimensions: tuple[int, int]
    scale_xy: tuple[float, float]
    padding_ltrb: tuple[int, int, int, int]
    interpolation: str
    crop: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "matrix": [list(row) for row in self.matrix],
            "source_dimensions_px": list(self.source_dimensions),
            "rotated_dimensions_px": list(self.rotated_dimensions),
            "resized_dimensions_px": list(self.resized_dimensions),
            "destination_dimensions_px": list(self.destination_dimensions),
            "scale_xy": list(self.scale_xy),
            "padding_ltrb_px": list(self.padding_ltrb),
            "interpolation": self.interpolation,
            "crop": self.crop,
        }


@dataclass(frozen=True, slots=True, eq=False)
class InputNormalizationResult:
    """Canonical page and auditable evidence for the user-confirmed input path."""

    canonical_rgb: np.ndarray
    input_mode: InputMode
    source_pixel_dimensions: tuple[int, int]
    source_physical_dimensions_mm: tuple[float, float] | None
    requested_orientation: OrientationChoice | None
    selected_rotation: OrientationChoice
    detected_orientation: str
    applied_transform: AppliedTransform
    canonical_dimensions: tuple[int, int]
    confidence: float
    confirmation_required: bool
    warnings: tuple[str, ...]
    marker_calibration_used: bool
    marker_ids: tuple[int, ...] = ()
    physical_coordinates_calibrated: bool = True
    pdf_rotation_degrees: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "accepted": True,
            "input_mode": self.input_mode.value,
            "source_pixel_dimensions_px": list(self.source_pixel_dimensions),
            "source_physical_dimensions_mm": (
                None
                if self.source_physical_dimensions_mm is None
                else list(self.source_physical_dimensions_mm)
            ),
            "requested_orientation": (
                None if self.requested_orientation is None else self.requested_orientation.value
            ),
            "selected_rotation": self.selected_rotation.value,
            "detected_orientation": self.detected_orientation,
            "applied_transform": self.applied_transform.to_dict(),
            "canonical_dimensions_px": list(self.canonical_dimensions),
            "canonical_colour_space": "RGB",
            "confidence": self.confidence,
            "confirmation_required": self.confirmation_required,
            "warnings": list(self.warnings),
            "marker_calibration_used": self.marker_calibration_used,
            "detected_ids": list(self.marker_ids),
            "physical_coordinates_calibrated": self.physical_coordinates_calibrated,
            "pdf_rotation_degrees": self.pdf_rotation_degrees,
        }


def _as_rgb(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim == 2:
        rgb = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    elif pixels.ndim == 3 and pixels.shape[2] == 3:
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    elif pixels.ndim == 3 and pixels.shape[2] == 4:
        rgb = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"unsupported input image shape: {pixels.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"input image must have dtype uint8, got {rgb.dtype}")
    return np.ascontiguousarray(rgb)


def _relative_a4_aspect_error(dimensions: tuple[float, float]) -> float:
    width, height = dimensions
    short, long = sorted((float(width), float(height)))
    if short <= 0:
        return float("inf")
    return abs(short / long - 210.0 / 297.0) / (210.0 / 297.0)


def _physical_is_a4(
    dimensions: tuple[float, float], tolerance: ScannerA4Tolerance
) -> bool:
    short, long = sorted(dimensions)
    return (
        abs(short - 210.0) <= tolerance.physical_mm
        and abs(long - 297.0) <= tolerance.physical_mm
        and _relative_a4_aspect_error(dimensions) <= tolerance.aspect_ratio_relative
    )


def _pixels_are_300dpi_a4(
    dimensions: tuple[int, int], layout: A4CalibrationLayout, tolerance: ScannerA4Tolerance
) -> bool:
    short, long = sorted(dimensions)
    expected_short, expected_long = sorted(layout.raster_size_px)
    return (
        abs(short - expected_short) <= tolerance.raster_dimension_px
        and abs(long - expected_long) <= tolerance.raster_dimension_px
        and _relative_a4_aspect_error(dimensions) <= tolerance.aspect_ratio_relative
    )


def _rotation_matrix(choice: OrientationChoice, width: int, height: int) -> tuple[np.ndarray, tuple[int, int]]:
    if choice is OrientationChoice.NO_ROTATION:
        return np.eye(3, dtype=np.float64), (width, height)
    if choice is OrientationChoice.ROTATE_LEFT_90:
        return np.asarray(((0, 1, 0), (-1, 0, width), (0, 0, 1)), dtype=np.float64), (height, width)
    if choice is OrientationChoice.ROTATE_RIGHT_90:
        return np.asarray(((0, -1, height), (1, 0, 0), (0, 0, 1)), dtype=np.float64), (height, width)
    if choice is OrientationChoice.ROTATE_180:
        return np.asarray(((-1, 0, width), (0, -1, height), (0, 0, 1)), dtype=np.float64), (width, height)
    raise ValueError(f"unresolved orientation choice: {choice}")


def _apply_rotation(rgb: np.ndarray, choice: OrientationChoice) -> np.ndarray:
    if choice is OrientationChoice.NO_ROTATION:
        return rgb.copy()
    if choice is OrientationChoice.ROTATE_LEFT_90:
        return np.ascontiguousarray(np.rot90(rgb, 1))
    if choice is OrientationChoice.ROTATE_RIGHT_90:
        return np.ascontiguousarray(np.rot90(rgb, 3))
    if choice is OrientationChoice.ROTATE_180:
        return np.ascontiguousarray(np.rot90(rgb, 2))
    raise ValueError(f"unresolved orientation choice: {choice}")


def _resolve_orientation(
    requested: OrientationChoice, source_dimensions: tuple[int, int]
) -> OrientationChoice:
    if requested is not OrientationChoice.AUTO:
        return requested
    width, height = source_dimensions
    # Canon imageRUNNER profile: landscape A4 raster with absent rotation metadata
    # is fed left-edge-first and becomes upright after a counter-clockwise turn.
    return OrientationChoice.ROTATE_LEFT_90 if width > height else OrientationChoice.NO_ROTATION


def _normalize_scanner_page(
    page: LoadedDocumentPage,
    *,
    orientation: OrientationChoice,
    layout: A4CalibrationLayout,
    tolerance: ScannerA4Tolerance,
) -> InputNormalizationResult:
    source = page.pixels
    source_copy = source.copy()
    source_dimensions = page.pixel_dimensions
    physical_match = (
        page.physical_dimensions_mm is not None
        and _physical_is_a4(page.physical_dimensions_mm, tolerance)
    )
    pixel_match = _pixels_are_300dpi_a4(source_dimensions, layout, tolerance)
    if not physical_match and not pixel_match:
        physical_text = (
            "unknown"
            if page.physical_dimensions_mm is None
            else f"{page.physical_dimensions_mm[0]:.1f} x {page.physical_dimensions_mm[1]:.1f} mm"
        )
        raise ValueError(
            "Scanner A4 mode requires a recognizable full-page A4 scan at 300 DPI; "
            f"received {source_dimensions[0]} x {source_dimensions[1]} px, physical size {physical_text}. "
            "No calibrated physical coordinates were assigned."
        )

    selected = _resolve_orientation(orientation, source_dimensions)
    source_rgb = _as_rgb(source)
    rotated = _apply_rotation(source_rgb, selected)
    width, height = source_dimensions
    rotation_matrix, rotated_dimensions = _rotation_matrix(selected, width, height)
    rotated_width, rotated_height = rotated_dimensions
    destination_width, destination_height = layout.raster_size_px
    portrait_error = abs(
        rotated_width / rotated_height - destination_width / destination_height
    ) / (destination_width / destination_height)
    warnings: list[str] = []

    if portrait_error <= tolerance.aspect_ratio_relative:
        resized_width, resized_height = destination_width, destination_height
        left = top = right = bottom = 0
        scale_x = resized_width / rotated_width
        scale_y = resized_height / rotated_height
        if (rotated_width, rotated_height) != (destination_width, destination_height):
            warnings.append(
                "The A4 raster was deterministically normalized from "
                f"{rotated_width} x {rotated_height} to {destination_width} x {destination_height} px; "
                "no content was cropped."
            )
        operation = "rotate_then_exact_a4_normalization"
    else:
        scale = min(destination_width / rotated_width, destination_height / rotated_height)
        resized_width = max(1, int(round(rotated_width * scale)))
        resized_height = max(1, int(round(rotated_height * scale)))
        left = (destination_width - resized_width) // 2
        top = (destination_height - resized_height) // 2
        right = destination_width - resized_width - left
        bottom = destination_height - resized_height - top
        scale_x = resized_width / rotated_width
        scale_y = resized_height / rotated_height
        operation = "rotate_then_fit_with_padding"
        warnings.append(
            "The selected orientation is landscape inside the portrait A4 workspace. "
            "It was fitted with white padding and no cropping; verify this override carefully."
        )

    interpolation_code = cv2.INTER_AREA if max(scale_x, scale_y) <= 1.0 else cv2.INTER_LINEAR
    interpolation_name = "INTER_AREA" if interpolation_code == cv2.INTER_AREA else "INTER_LINEAR"
    resized = cv2.resize(rotated, (resized_width, resized_height), interpolation=interpolation_code)
    canonical = np.full((destination_height, destination_width, 3), 255, dtype=np.uint8)
    canonical[top : top + resized_height, left : left + resized_width] = resized

    scale_translate = np.asarray(
        ((scale_x, 0, left), (0, scale_y, top), (0, 0, 1)), dtype=np.float64
    )
    matrix = scale_translate @ rotation_matrix
    if not np.array_equal(source, source_copy):
        raise RuntimeError("scanner normalization mutated the caller-owned source array")
    if page.physical_dimensions_mm is None:
        warnings.append(
            "A4 size was inferred from the nominal 300 DPI raster dimensions because no physical metadata was embedded."
        )
    elif not physical_match:
        warnings.append(
            "Embedded physical-size metadata did not describe A4; acceptance used the nominal 300 DPI A4 raster dimensions."
        )

    detected = "landscape A4" if width > height else "portrait A4"
    transform = AppliedTransform(
        operation=operation,
        matrix=tuple(tuple(float(value) for value in row) for row in matrix),
        source_dimensions=source_dimensions,
        rotated_dimensions=rotated_dimensions,
        resized_dimensions=(resized_width, resized_height),
        destination_dimensions=layout.raster_size_px,
        scale_xy=(scale_x, scale_y),
        padding_ltrb=(left, top, right, bottom),
        interpolation=interpolation_name,
    )
    return InputNormalizationResult(
        canonical_rgb=np.ascontiguousarray(canonical),
        input_mode=InputMode.SCANNER_A4,
        source_pixel_dimensions=source_dimensions,
        source_physical_dimensions_mm=page.physical_dimensions_mm,
        requested_orientation=orientation,
        selected_rotation=selected,
        detected_orientation=detected,
        applied_transform=transform,
        canonical_dimensions=layout.raster_size_px,
        confidence=0.99 if physical_match else (0.80 if page.physical_dimensions_mm else 0.88),
        confirmation_required=True,
        warnings=tuple(warnings),
        marker_calibration_used=False,
        physical_coordinates_calibrated=(portrait_error <= tolerance.aspect_ratio_relative),
        pdf_rotation_degrees=page.pdf_rotation_degrees,
    )


def _normalize_marked_page(
    page: LoadedDocumentPage, *, layout: A4CalibrationLayout
) -> InputNormalizationResult:
    detection = detect_and_rectify(page.pixels, layout)
    if not detection.accepted or detection.warped is None or detection.source_to_canonical is None:
        raise ValueError(f"page alignment rejected: {detection.reason}")
    canonical = _as_rgb(detection.warped)
    transform = AppliedTransform(
        operation="aruco_perspective_rectification",
        matrix=tuple(
            tuple(float(value) for value in row) for row in detection.source_to_canonical
        ),
        source_dimensions=page.pixel_dimensions,
        rotated_dimensions=page.pixel_dimensions,
        resized_dimensions=layout.raster_size_px,
        destination_dimensions=layout.raster_size_px,
        scale_xy=(1.0, 1.0),
        padding_ltrb=(0, 0, 0, 0),
        interpolation="INTER_LINEAR",
    )
    return InputNormalizationResult(
        canonical_rgb=canonical,
        input_mode=InputMode.MARKED_PAGE,
        source_pixel_dimensions=page.pixel_dimensions,
        source_physical_dimensions_mm=page.physical_dimensions_mm,
        requested_orientation=None,
        selected_rotation=OrientationChoice.NO_ROTATION,
        detected_orientation=f"{detection.orientation_degrees or 0}° from ArUco markers",
        applied_transform=transform,
        canonical_dimensions=layout.raster_size_px,
        confidence=detection.confidence,
        confirmation_required=True,
        warnings=tuple(detection.warnings),
        marker_calibration_used=True,
        marker_ids=detection.detected_ids,
        physical_coordinates_calibrated=True,
        pdf_rotation_degrees=page.pdf_rotation_degrees,
    )


def normalize_document_input(
    input_path: str | Path,
    *,
    input_mode: InputMode | str,
    orientation: OrientationChoice | str = OrientationChoice.AUTO,
    page_index: int = 0,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    tolerance: ScannerA4Tolerance = ScannerA4Tolerance(),
) -> InputNormalizationResult:
    """Normalize one document using only the explicitly selected input mode."""
    mode = InputMode(input_mode)
    choice = OrientationChoice(orientation)
    page = load_document_page_info(
        input_path, page_index=page_index, dpi=layout.dpi, grayscale=False
    )
    if mode is InputMode.SCANNER_A4:
        return _normalize_scanner_page(
            page, orientation=choice, layout=layout, tolerance=tolerance
        )
    return _normalize_marked_page(page, layout=layout)


__all__ = [
    "AppliedTransform",
    "InputMode",
    "InputNormalizationResult",
    "OrientationChoice",
    "ScannerA4Tolerance",
    "normalize_document_input",
]
