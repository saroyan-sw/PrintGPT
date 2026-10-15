"""Load raster images and PDF pages into OpenCV-compatible arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium

from .geometry import DEFAULT_DPI


def _validate_options(page_index: int, dpi: float) -> None:
    if page_index < 0:
        raise ValueError("page_index must be zero or greater")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a positive finite number")


def _load_pdf_page(path: Path, page_index: int, dpi: float, grayscale: bool) -> np.ndarray:
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

    if grayscale:
        return pixels
    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)


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
    _validate_options(page_index, dpi)
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        return _load_pdf_page(path, page_index, dpi, grayscale)

    if page_index != 0:
        raise ValueError("raster image inputs contain only page 1")
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None or image.size == 0:
        raise ValueError(f"could not decode image {path}")
    return image
