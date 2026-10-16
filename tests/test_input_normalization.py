from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from printgpt_alignment import LoadedDocumentPage
from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_alignment.geometry import A4CalibrationLayout, DEFAULT_LAYOUT
from printgpt_mvp import (
    InputMode,
    OrientationChoice,
    ScannerA4Tolerance,
    normalize_document_input,
)


LAYOUT = A4CalibrationLayout(dpi=75.0)


def _portrait_page() -> np.ndarray:
    page = np.full((*LAYOUT.image_shape, 3), 255, dtype=np.uint8)
    cv2.putText(page, "UPRIGHT", (60, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    cv2.arrowedLine(page, (100, 500), (100, 250), (0, 0, 0), 8, tipLength=0.15)
    return page


def _loaded(pixels: np.ndarray, physical=None) -> LoadedDocumentPage:
    height, width = pixels.shape[:2]
    return LoadedDocumentPage(
        pixels=pixels,
        pixel_dimensions=(width, height),
        physical_dimensions_mm=physical,
        source_dpi=None,
        pdf_rotation_degrees=0,
        is_pdf=False,
    )


def _normalize_loaded(page: LoadedDocumentPage, orientation=OrientationChoice.AUTO):
    with patch("printgpt_mvp.input_normalization.load_document_page_info", return_value=page):
        return normalize_document_input(
            "unused.png",
            input_mode=InputMode.SCANNER_A4,
            orientation=orientation,
            layout=LAYOUT,
        )


def test_auto_rotates_canon_landscape_left_and_preserves_source_array():
    upright = _portrait_page()
    landscape = np.ascontiguousarray(np.rot90(upright, 3))
    original = landscape.copy()

    result = _normalize_loaded(_loaded(landscape, (297.0, 210.0)))

    assert result.selected_rotation is OrientationChoice.ROTATE_LEFT_90
    assert result.detected_orientation == "landscape A4"
    assert result.canonical_rgb.shape == (*LAYOUT.image_shape, 3)
    np.testing.assert_array_equal(result.canonical_rgb, cv2.cvtColor(upright, cv2.COLOR_BGR2RGB))
    np.testing.assert_array_equal(landscape, original)
    assert result.applied_transform.crop is None
    assert result.applied_transform.padding_ltrb == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (OrientationChoice.ROTATE_LEFT_90, OrientationChoice.ROTATE_LEFT_90),
        (OrientationChoice.ROTATE_RIGHT_90, OrientationChoice.ROTATE_RIGHT_90),
        (OrientationChoice.ROTATE_180, OrientationChoice.ROTATE_180),
        (OrientationChoice.NO_ROTATION, OrientationChoice.NO_ROTATION),
    ],
)
def test_manual_orientation_overrides_are_recorded(choice, expected):
    landscape = np.ascontiguousarray(np.rot90(_portrait_page(), 3))
    result = _normalize_loaded(_loaded(landscape, (297.0, 210.0)), choice)
    assert result.requested_orientation is choice
    assert result.selected_rotation is expected
    assert result.canonical_rgb.shape == (*LAYOUT.image_shape, 3)
    if choice in (OrientationChoice.NO_ROTATION, OrientationChoice.ROTATE_180):
        assert any(result.applied_transform.padding_ltrb)
        assert not result.physical_coordinates_calibrated


def test_one_pixel_a4_difference_is_normalized_without_crop():
    width, height = LAYOUT.raster_size_px
    page = np.full((height - 1, width + 1, 3), 255, dtype=np.uint8)
    result = _normalize_loaded(_loaded(page))
    assert result.canonical_rgb.shape == (height, width, 3)
    assert result.applied_transform.resized_dimensions == (width, height)
    assert result.applied_transform.crop is None
    assert "no content was cropped" in result.warnings[0]


def test_a4_tolerance_is_configurable_and_non_a4_is_rejected():
    square = _loaded(np.full((600, 600, 3), 255, dtype=np.uint8))
    with patch("printgpt_mvp.input_normalization.load_document_page_info", return_value=square):
        with pytest.raises(ValueError, match="recognizable full-page A4"):
            normalize_document_input(
                "unused.png",
                input_mode=InputMode.SCANNER_A4,
                layout=LAYOUT,
                tolerance=ScannerA4Tolerance(raster_dimension_px=500),
            )


def test_marked_mode_is_strict_and_reports_marker_confidence(tmp_path):
    marked_path = tmp_path / "marked.png"
    assert cv2.imwrite(str(marked_path), render_calibration_image(LAYOUT))
    marked = normalize_document_input(
        marked_path, input_mode=InputMode.MARKED_PAGE, layout=LAYOUT
    )
    assert marked.input_mode is InputMode.MARKED_PAGE
    assert marked.marker_calibration_used
    assert set(marked.marker_ids) >= set(LAYOUT.expected_marker_ids)
    assert marked.confidence >= 0.60

    unmarked_path = tmp_path / "unmarked.png"
    assert cv2.imwrite(str(unmarked_path), _portrait_page())
    with pytest.raises(ValueError, match="no ArUco markers"):
        normalize_document_input(
            unmarked_path, input_mode=InputMode.MARKED_PAGE, layout=LAYOUT
        )


def test_real_canon_sample_when_available():
    path = Path(r"C:\Users\saroy\OneDrive\Documents\Scans\0775_001.pdf")
    if not path.is_file():
        pytest.skip("local Canon scanner sample is not available")
    result = normalize_document_input(path, input_mode=InputMode.SCANNER_A4)
    assert result.source_pixel_dimensions == (3507, 2481)
    assert result.selected_rotation is OrientationChoice.ROTATE_LEFT_90
    assert result.canonical_rgb.shape == (3508, 2480, 3)
    assert result.applied_transform.resized_dimensions == (2480, 3508)
    assert result.applied_transform.padding_ltrb == (0, 0, 0, 0)
    assert result.applied_transform.crop is None
