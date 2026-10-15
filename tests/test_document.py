from pathlib import Path

import cv2
import numpy as np
import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from printgpt_alignment.calibration_page import write_calibration_pdf
from printgpt_alignment.detection import detect_and_rectify
from printgpt_alignment.document import load_document_page
from printgpt_alignment.geometry import DEFAULT_LAYOUT


def test_loads_raster_image(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    source = np.full((31, 47, 3), (12, 34, 56), dtype=np.uint8)
    assert cv2.imwrite(str(path), source)

    loaded = load_document_page(path)

    assert loaded.shape == source.shape
    assert np.array_equal(loaded, source)


def test_rejects_nonzero_page_for_raster(tmp_path: Path) -> None:
    path = tmp_path / "sample.png"
    assert cv2.imwrite(str(path), np.full((10, 10), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="only page 1"):
        load_document_page(path, page_index=1)


def test_loads_selected_pdf_page_at_requested_dpi(tmp_path: Path) -> None:
    path = tmp_path / "two_pages.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFillGray(1.0)
    pdf.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    pdf.showPage()
    pdf.setFillGray(0.0)
    pdf.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)
    pdf.showPage()
    pdf.save()

    first = load_document_page(path, page_index=0, dpi=144, grayscale=True)
    second = load_document_page(path, page_index=1, dpi=144, grayscale=True)

    expected_width = round(210.0 * 144 / 25.4)
    expected_height = round(297.0 * 144 / 25.4)
    assert abs(first.shape[1] - expected_width) <= 1
    assert abs(first.shape[0] - expected_height) <= 1
    assert float(first.mean()) > 250
    assert float(second.mean()) < 5


def test_pdf_page_out_of_range_reports_page_count(tmp_path: Path) -> None:
    path = tmp_path / "one_page.pdf"
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.showPage()
    pdf.save()

    with pytest.raises(IndexError, match=r"page 2.*has 1 page"):
        load_document_page(path, page_index=1)


def test_calibration_pdf_round_trip_is_detectable(tmp_path: Path) -> None:
    path = write_calibration_pdf(tmp_path / "calibration.pdf")

    page = load_document_page(path, dpi=DEFAULT_LAYOUT.dpi, grayscale=True)
    result = detect_and_rectify(page)

    assert abs(page.shape[1] - DEFAULT_LAYOUT.raster_width_px) <= 1
    assert abs(page.shape[0] - DEFAULT_LAYOUT.raster_height_px) <= 1
    assert result.accepted, result.reason
    assert result.warped is not None
    assert result.warped.shape == DEFAULT_LAYOUT.image_shape
