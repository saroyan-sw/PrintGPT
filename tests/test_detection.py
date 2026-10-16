import cv2
import numpy as np
import pytest

from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_alignment.detection import detect_and_rectify, detect_and_rectify_document
from printgpt_alignment.geometry import A4CalibrationLayout


def _perspective_capture(layout: A4CalibrationLayout) -> tuple[np.ndarray, np.ndarray]:
    page = render_calibration_image(layout)
    canvas_size = (1300, 1800)
    destination = np.asarray([[160, 115], [1125, 170], [1170, 1630], [105, 1680]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(np.asarray(layout.page_corners_px, dtype=np.float32), destination)
    capture = cv2.warpPerspective(page, homography, canvas_size, borderValue=205)
    return capture, homography


def test_detects_and_rectifies_perspective_page() -> None:
    layout = A4CalibrationLayout(dpi=150.0)
    capture, _ = _perspective_capture(layout)
    result = detect_and_rectify(capture, layout)
    assert result.accepted, result.reason
    assert result.warped is not None
    assert result.warped.shape == layout.image_shape
    assert result.orientation_degrees == 0


def test_identifies_clockwise_orientation() -> None:
    layout = A4CalibrationLayout(dpi=150.0)
    image = render_calibration_image(layout)
    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    result = detect_and_rectify(rotated, layout)
    assert result.accepted, result.reason
    assert result.orientation_degrees == 90


def test_rejects_missing_marker() -> None:
    layout = A4CalibrationLayout(dpi=150.0)
    image = render_calibration_image(layout)
    marker = layout.marker_by_id(10)
    x0, y0 = marker.bounds_mm.corners[0].to_pixels(layout.dpi)
    x1, y1 = marker.bounds_mm.corners[2].to_pixels(layout.dpi)
    cv2.rectangle(image, (int(x0) - 5, int(y0) - 5), (int(x1) + 5, int(y1) + 5), 255, -1)
    result = detect_and_rectify(image, layout)
    assert not result.accepted
    assert "missing" in result.reason


def _unmarked_page(layout: A4CalibrationLayout) -> np.ndarray:
    page = np.full(layout.image_shape, 255, dtype=np.uint8)
    cv2.putText(
        page,
        "PrintGPT marker-free scan",
        (50, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(page, (80, 180), (layout.raster_width_px - 80, 420), 0, 3)
    cv2.line(page, (100, 520), (layout.raster_width_px - 100, 650), 0, 4)
    return page


def test_rectifies_unmarked_page_from_document_boundary() -> None:
    layout = A4CalibrationLayout(dpi=75.0)
    page = _unmarked_page(layout)
    canvas = np.full((1200, 1000), 55, dtype=np.uint8)
    destination = np.asarray(
        [[170, 90], [810, 130], [850, 1060], [120, 1090]], dtype=np.float32
    )
    transform = cv2.getPerspectiveTransform(
        np.asarray(layout.page_corners_px, dtype=np.float32), destination
    )
    capture = cv2.warpPerspective(
        page,
        transform,
        (1000, 1200),
        dst=canvas,
        borderMode=cv2.BORDER_TRANSPARENT,
    )

    result = detect_and_rectify_document(capture, layout)

    assert result.accepted, result.reason
    assert result.alignment_method == "page_boundary"
    assert result.detected_ids == ()
    assert result.source_page_corners is not None
    assert result.warped is not None and result.warped.shape == layout.image_shape


def test_accepts_clean_full_frame_unmarked_scan() -> None:
    layout = A4CalibrationLayout(dpi=75.0)
    result = detect_and_rectify_document(_unmarked_page(layout), layout)
    assert result.accepted, result.reason
    assert result.alignment_method == "scan_frame"
    assert result.page_area_ratio == 1.0


def test_bad_marker_free_scan_explicitly_requests_rescan() -> None:
    layout = A4CalibrationLayout(dpi=75.0)
    unusable = np.full(layout.image_shape, 245, dtype=np.uint8)
    result = detect_and_rectify_document(unusable, layout)
    assert not result.accepted
    assert "scan again" in result.reason.lower()
    assert "contrast" in result.reason.lower() or "blurry" in result.reason.lower()


@pytest.mark.parametrize("width", [754, 767, 802, 1000])
@pytest.mark.parametrize("scanner_bars", [False, True])
def test_scanner_exports_do_not_require_a4_proportions_or_external_edges(width, scanner_bars):
    layout = A4CalibrationLayout(dpi=75)
    page = cv2.resize(_unmarked_page(layout), (width, 1000))
    if scanner_bars:
        page[10:35, 5:-5] = 0
        page[-40:-10, 5:-5] = 0
    result = detect_and_rectify_document(page, layout)
    assert result.accepted, result.reason
    assert result.alignment_method == "scan_frame"
    np.testing.assert_allclose(result.source_page_corners, [[0, 0], [width, 0], [width, 1000], [0, 1000]])
    transform = result.source_to_canonical
    assert transform[0, 0] == pytest.approx(transform[1, 1], rel=1e-6)
    assert result.warnings  # Can't infer physical A4 from these pixel dimensions.
    assert result.warped.shape == layout.image_shape


def test_a4_shaped_photo_is_not_mistaken_for_full_frame_scan():
    layout = A4CalibrationLayout(dpi=75)
    photo = np.full((1400, 990), 45, dtype=np.uint8)
    page = cv2.resize(_unmarked_page(layout), (700, 990))
    photo[205:1195, 145:845] = page
    result = detect_and_rectify_document(photo, layout)
    assert result.accepted, result.reason
    assert result.alignment_method == "page_boundary"
    assert result.page_area_ratio < 0.8


@pytest.mark.parametrize("input_kind", ["auto", "scan"])
@pytest.mark.parametrize("degradation", ["blur", "low_contrast", "small"])
def test_bad_scans_still_require_rescan(input_kind, degradation):
    layout = A4CalibrationLayout(dpi=75)
    page = _unmarked_page(layout)
    if degradation == "blur":
        page = cv2.GaussianBlur(page, (51, 51), 12)
    elif degradation == "low_contrast":
        page = (245 + page.astype(np.float32) / 255 * 8).astype(np.uint8)
    else:
        page = cv2.resize(page, (160, 220))
    result = detect_and_rectify_document(page, layout, input_kind=input_kind)
    assert not result.accepted
    assert result.warped is None
    assert "scan again" in result.reason.lower()


def test_photo_mode_does_not_fall_back_to_image_bounds():
    layout = A4CalibrationLayout(dpi=75)
    result = detect_and_rectify_document(_unmarked_page(layout), layout, input_kind="photo")
    assert not result.accepted
    assert "Scanned page" in result.reason


def test_general_document_entrypoint_preserves_notebook_marker_alignment():
    layout = A4CalibrationLayout(dpi=150)
    page, _ = _perspective_capture(layout)
    expected = detect_and_rectify(page, layout)
    actual = detect_and_rectify_document(page, layout)
    assert actual.accepted
    assert actual.alignment_method == "aruco"
    np.testing.assert_allclose(actual.source_to_canonical, expected.source_to_canonical)
    np.testing.assert_array_equal(actual.warped, expected.warped)
