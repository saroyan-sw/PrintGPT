"""Render the versioned PrintGPT A4 calibration page as PNG and PDF."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from reportlab.lib.colors import Color, black, white
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdf_canvas

from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_pixel_index


def aruco_dictionary(layout: A4CalibrationLayout = DEFAULT_LAYOUT):
    try:
        dictionary_id = getattr(cv2.aruco, layout.dictionary_name)
    except AttributeError as exc:
        raise ValueError(f"unknown ArUco dictionary: {layout.dictionary_name}") from exc
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def marker_modules(marker_id: int, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """Return the exact black/white module grid, including the black border."""
    dictionary = aruco_dictionary(layout)
    marker_bits = int(dictionary.markerSize)
    side_modules = marker_bits + 2
    return cv2.aruco.generateImageMarker(dictionary, marker_id, side_modules, borderBits=1)


def _draw_dashed_rectangle(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> None:
    color = 180
    thickness = max(1, round(min(image.shape[:2]) / 1800))
    dash = max(8, round(min(image.shape[:2]) / 120))
    gap = max(5, dash // 2)
    for start in range(x0, x1, dash + gap):
        cv2.line(image, (start, y0), (min(start + dash, x1), y0), color, thickness)
        cv2.line(image, (start, y1), (min(start + dash, x1), y1), color, thickness)
    for start in range(y0, y1, dash + gap):
        cv2.line(image, (x0, start), (x0, min(start + dash, y1)), color, thickness)
        cv2.line(image, (x1, start), (x1, min(start + dash, y1)), color, thickness)


def _fit_background(image: np.ndarray, background: np.ndarray, layout: A4CalibrationLayout) -> None:
    if background.ndim == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    area = layout.protected_content_mm
    x0 = mm_to_pixel_index(area.x, layout.dpi)
    y0 = mm_to_pixel_index(area.y, layout.dpi)
    x1 = mm_to_pixel_index(area.right, layout.dpi)
    y1 = mm_to_pixel_index(area.bottom, layout.dpi)
    target_w, target_h = x1 - x0, y1 - y0
    scale = min(target_w / background.shape[1], target_h / background.shape[0])
    width = max(1, round(background.shape[1] * scale))
    height = max(1, round(background.shape[0] * scale))
    resized = cv2.resize(background, (width, height), interpolation=cv2.INTER_AREA)
    ox = x0 + (target_w - width) // 2
    oy = y0 + (target_h - height) // 2
    image[oy : oy + height, ox : ox + width] = resized


def render_calibration_image(
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Render a canonical grayscale calibration raster."""
    image = np.full(layout.image_shape, 255, dtype=np.uint8)
    if background is not None:
        _fit_background(image, background, layout)

    margin = layout.protected_margin_mm
    x0 = mm_to_pixel_index(margin, layout.dpi)
    y0 = mm_to_pixel_index(margin, layout.dpi)
    x1 = mm_to_pixel_index(layout.page_width_mm - margin, layout.dpi)
    y1 = mm_to_pixel_index(layout.page_height_mm - margin, layout.dpi)
    _draw_dashed_rectangle(image, x0, y0, x1, y1)

    for marker in layout.markers:
        quiet = layout.marker_quiet_zone_mm
        bounds = marker.bounds_mm
        qx0 = max(0, mm_to_pixel_index(bounds.x - quiet, layout.dpi))
        qy0 = max(0, mm_to_pixel_index(bounds.y - quiet, layout.dpi))
        qx1 = min(layout.raster_width_px, mm_to_pixel_index(bounds.right + quiet, layout.dpi))
        qy1 = min(layout.raster_height_px, mm_to_pixel_index(bounds.bottom + quiet, layout.dpi))
        image[qy0:qy1, qx0:qx1] = 255
        marker_size = mm_to_pixel_index(layout.marker_size_mm, layout.dpi)
        marker_image = cv2.aruco.generateImageMarker(
            aruco_dictionary(layout), marker.marker_id, marker_size, borderBits=1
        )
        mx = mm_to_pixel_index(bounds.x, layout.dpi)
        my = mm_to_pixel_index(bounds.y, layout.dpi)
        image[my : my + marker_size, mx : mx + marker_size] = marker_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = layout.dpi / 300.0
    center_x = layout.raster_width_px // 2
    lines = [
        ("^  TOP  ^", 16.0, 0.72 * scale, 2),
        (layout.layout_id, 22.0, 0.48 * scale, 1),
        ("PRINT AT ACTUAL SIZE - 100% - DO NOT FIT TO PAGE", 279.0, 0.42 * scale, 1),
        ("Protected content boundary: 10 mm", 284.0, 0.36 * scale, 1),
    ]
    for text, y_mm, font_scale, thickness in lines:
        (text_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        y = mm_to_pixel_index(y_mm, layout.dpi)
        cv2.putText(image, text, (center_x - text_w // 2, y), font, font_scale, 40, thickness, cv2.LINE_AA)
    return image


def write_calibration_png(
    output_path: str | Path,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), render_calibration_image(layout)):
        raise OSError(f"failed to write {output}")
    return output


def _pdf_y(page_height_mm: float, top_y_mm: float) -> float:
    return (page_height_mm - top_y_mm) * mm


def write_calibration_pdf(
    output_path: str | Path,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
) -> Path:
    """Write a true-size A4 vector PDF. Print with scaling disabled."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    page_size = (layout.page_width_mm * mm, layout.page_height_mm * mm)
    canvas = pdf_canvas.Canvas(str(output), pagesize=page_size, pageCompression=1)
    canvas.setTitle(layout.layout_id)
    canvas.setAuthor("PrintGPT")

    margin = layout.protected_margin_mm
    canvas.setStrokeColor(Color(0.65, 0.65, 0.65))
    canvas.setLineWidth(0.25 * mm)
    canvas.setDash(3 * mm, 2 * mm)
    canvas.rect(
        margin * mm,
        margin * mm,
        (layout.page_width_mm - 2 * margin) * mm,
        (layout.page_height_mm - 2 * margin) * mm,
        stroke=1,
        fill=0,
    )
    canvas.setDash()

    for marker in layout.markers:
        bounds = marker.bounds_mm
        quiet = layout.marker_quiet_zone_mm
        canvas.setFillColor(white)
        canvas.rect(
            (bounds.x - quiet) * mm,
            _pdf_y(layout.page_height_mm, bounds.bottom + quiet),
            (bounds.width + 2 * quiet) * mm,
            (bounds.height + 2 * quiet) * mm,
            stroke=0,
            fill=1,
        )
        modules = marker_modules(marker.marker_id, layout)
        module_mm = bounds.width / modules.shape[1]
        canvas.setFillColor(black)
        for row in range(modules.shape[0]):
            for col in range(modules.shape[1]):
                if modules[row, col] == 0:
                    x = bounds.x + col * module_mm
                    top_y = bounds.y + row * module_mm
                    canvas.rect(
                        x * mm,
                        _pdf_y(layout.page_height_mm, top_y + module_mm),
                        module_mm * mm,
                        module_mm * mm,
                        stroke=0,
                        fill=1,
                    )

    canvas.setFillColor(black)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(layout.page_width_mm * mm / 2, _pdf_y(layout.page_height_mm, 16), "^  TOP  ^")
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(layout.page_width_mm * mm / 2, _pdf_y(layout.page_height_mm, 22), layout.layout_id)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawCentredString(layout.page_width_mm * mm / 2, _pdf_y(layout.page_height_mm, 279), "PRINT AT ACTUAL SIZE - 100% - DO NOT FIT TO PAGE")
    canvas.drawCentredString(layout.page_width_mm * mm / 2, _pdf_y(layout.page_height_mm, 284), "Protected content boundary: 10 mm")
    canvas.setFont("Helvetica", 5.5)
    for marker in layout.markers:
        x = (marker.bounds_mm.x + marker.bounds_mm.width / 2) * mm
        if marker.role.startswith("top"):
            y = _pdf_y(layout.page_height_mm, marker.bounds_mm.bottom + 5.0)
        else:
            y = _pdf_y(layout.page_height_mm, marker.bounds_mm.y - 4.0)
        canvas.drawCentredString(x, y, f"ID {marker.marker_id}")

    canvas.showPage()
    canvas.save()
    return output
