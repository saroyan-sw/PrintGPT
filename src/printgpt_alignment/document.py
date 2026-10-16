"""Load raster images and PDF pages into OpenCV-compatible arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from .geometry import DEFAULT_DPI, MM_PER_INCH


@dataclass(frozen=True, slots=True, eq=False)
class LoadedDocumentPage:
    """Rendered page pixels plus trustworthy source geometry when available."""

    pixels: np.ndarray
    pixel_dimensions: tuple[int, int]
    physical_dimensions_mm: tuple[float, float] | None
    source_dpi: tuple[float, float] | None
    pdf_rotation_degrees: int | None
    is_pdf: bool


def _validate_options(page_index: int, dpi: float) -> None:
    if page_index < 0:
        raise ValueError("page_index must be zero or greater")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a positive finite number")


def _load_pdf_page(
    path: Path, page_index: int, dpi: float, grayscale: bool
) -> LoadedDocumentPage:
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise ValueError(f"could not open PDF {path}: {exc}") from exc

    try:
        page_count = len(document)
        if page_index >= page_count:
            raise IndexError(
                f"PDF page {page_index + 1} does not exist; {path.name} has {page_count} page(s)"
            )

        page = document[page_index]
        try:
            width_pt, height_pt = page.get_size()
            rotation = int(page.get_rotation()) if hasattr(page, "get_rotation") else None
            bitmap = page.render(scale=float(dpi) / 72.0)
            try:
                mode = "L" if grayscale else "RGB"
                pixels = np.asarray(bitmap.to_pil().convert(mode)).copy()
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()

    if not grayscale:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    height_px, width_px = pixels.shape[:2]
    return LoadedDocumentPage(
        pixels=pixels,
        pixel_dimensions=(width_px, height_px),
        physical_dimensions_mm=(
            float(width_pt) * MM_PER_INCH / 72.0,
            float(height_pt) * MM_PER_INCH / 72.0,
        ),
        source_dpi=(float(dpi), float(dpi)),
        pdf_rotation_degrees=rotation,
        is_pdf=True,
    )


def _raster_metadata(path: Path, width_px: int, height_px: int) -> tuple[
    tuple[float, float] | None, tuple[float, float] | None
]:
    """Read optional raster DPI metadata without duplicating pixel decoding."""
    try:
        with Image.open(path) as image:
            raw_dpi = image.info.get("dpi")
    except Exception:
        return None, None
    if not isinstance(raw_dpi, (tuple, list)) or len(raw_dpi) < 2:
        return None, None
    try:
        dpi_x, dpi_y = float(raw_dpi[0]), float(raw_dpi[1])
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite((dpi_x, dpi_y)).all() or min(dpi_x, dpi_y) <= 0:
        return None, None
    return (
        (width_px * MM_PER_INCH / dpi_x, height_px * MM_PER_INCH / dpi_y),
        (dpi_x, dpi_y),
    )


def load_document_page_info(
    input_path: str | Path,
    *,
    page_index: int = 0,
    dpi: float = DEFAULT_DPI,
    grayscale: bool = False,
) -> LoadedDocumentPage:
    """Load pixels and source geometry through the shared document renderer."""
    _validate_options(page_index, dpi)
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        return _load_pdf_page(path, page_index, dpi, grayscale)

    if page_index != 0:
        raise ValueError("raster image inputs contain only page 1")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    pixels = cv2.imread(str(path), flag)
    if pixels is None or pixels.size == 0:
        raise ValueError(f"could not decode image {path}")
    height_px, width_px = pixels.shape[:2]
    physical, source_dpi = _raster_metadata(path, width_px, height_px)
    return LoadedDocumentPage(
        pixels=pixels,
        pixel_dimensions=(width_px, height_px),
        physical_dimensions_mm=physical,
        source_dpi=source_dpi,
        pdf_rotation_degrees=None,
        is_pdf=False,
    )


def load_document_page(
    input_path: str | Path,
    *,
    page_index: int = 0,
    dpi: float = DEFAULT_DPI,
    grayscale: bool = False,
) -> np.ndarray:
    """Load one image or PDF page as a NumPy array.

    ``page_index`` is zero-based in this Python API. PDF pages are rendered at
    ``dpi`` before entering the OpenCV alignment pipeline. Raster inputs are
    read directly and only support page index zero.
    """
    return load_document_page_info(
        input_path, page_index=page_index, dpi=dpi, grayscale=grayscale
    ).pixels
