from pathlib import Path

import cv2

from printgpt_alignment.calibration_page import render_calibration_image, write_calibration_pdf, write_calibration_png
from printgpt_alignment.geometry import A4CalibrationLayout


def test_calibration_raster_has_expected_markers() -> None:
    layout = A4CalibrationLayout(dpi=150.0)
    image = render_calibration_image(layout)
    assert image.shape == layout.image_shape
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        cv2.aruco.DetectorParameters(),
    )
    _, ids, _ = detector.detectMarkers(image)
    assert ids is not None
    assert set(int(value) for value in ids.ravel()) == {10, 11, 12, 13}


def test_writers_create_png_and_pdf(tmp_path: Path) -> None:
    layout = A4CalibrationLayout(dpi=100.0)
    png_path = write_calibration_png(tmp_path / "calibration.png", layout)
    pdf_path = write_calibration_pdf(tmp_path / "calibration.pdf", layout)
    assert cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE).shape == layout.image_shape
    assert pdf_path.read_bytes().startswith(b"%PDF")
