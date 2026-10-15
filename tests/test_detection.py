import cv2
import numpy as np

from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_alignment.detection import detect_and_rectify
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
