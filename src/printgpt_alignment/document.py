"""Load pixels without losing the evidence needed for physical registration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps
from pypdf import PdfReader

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


@dataclass(frozen=True)
class DocumentPage:
    image: np.ndarray
    metadata: dict[str, Any]


def load_document(
    input_path: str | Path, *, page_index: int = 0,
    dpi: float = DEFAULT_DPI, grayscale: bool = False,
) -> DocumentPage:
    """Pixels are BGR (or gray); coordinates use top-left pixel *edges*.

    PDF coordinates refer to the unrotated MediaBox, not the visible CropBox.
    Metadata describes a file, not proof that a scanner preserved the original.
    """
    if page_index < 0:
        raise ValueError("page_index must be zero or greater")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be a positive finite number")
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    meta: dict[str, Any] = {"kind": "raster", "page_index": page_index}
    if path.suffix.lower() == ".pdf":
        try:
            document = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ValueError(f"could not open PDF {path}: {exc}") from exc
        try:
            if page_index >= len(document):
                raise IndexError(f"PDF page {page_index + 1} does not exist; {path.name} has {len(document)} page(s)")
            page = document[page_index]
            try:
                bitmap = page.render(scale=float(dpi) / 72)
                try:
                    pixels = np.asarray(bitmap.to_pil().convert("L" if grayscale else "RGB")).copy()
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
        meta.update(kind="pdf", render_dpi=float(dpi))
        try:
            with path.open("rb") as stream:
                pdf_page = PdfReader(stream).pages[page_index]
                media = np.asarray(pdf_page.mediabox, dtype=float)
                crop = np.asarray(pdf_page.cropbox, dtype=float)
                unit = float(pdf_page.get("/UserUnit", 1))
                rotation = int(pdf_page.rotation) % 360
            if not np.isfinite(np.r_[media, crop, unit]).all() or unit <= 0 or rotation not in (0, 90, 180, 270):
                raise ValueError("invalid PDF page geometry")
            l, b = np.maximum(media[:2], crop[:2])
            r, t = np.minimum(media[2:], crop[2:])
            if r <= l or t <= b or np.any(media[2:] <= media[:2]):
                raise ValueError("empty PDF page box")
            factor = unit * 25.4 / 72
            # Rendered TL/TR/BR/BL expressed in original, unrotated paper mm.
            corners = np.array([[l, t], [r, t], [r, b], [l, b]])
            corners = np.roll(corners, rotation // 90, axis=0)
            corners[:, 0] = (corners[:, 0] - media[0]) * factor
            corners[:, 1] = (media[3] - corners[:, 1]) * factor
            expected_ratio = (r-l)/(t-b) if rotation % 180 == 0 else (t-b)/(r-l)
            if abs(pixels.shape[1] / pixels.shape[0] / expected_ratio - 1) > 0.01:
                raise ValueError("PDF renderer and page boxes disagree")
            meta.update(
                media_box_pt=media.tolist(), crop_box_pt=crop.tolist(), user_unit=unit,
                rotation_degrees=rotation,
                page_size_mm=((media[2:] - media[:2]) * factor).tolist(),
                source_corners_mm=corners.tolist(),
            )
        except Exception as exc:
            # Keep unusual/malformed PDFs usable as previews, without fabricating scale.
            meta["geometry_warning"] = f"PDF physical metadata unavailable: {exc}"
    else:
        if page_index != 0:
            raise ValueError("raster image inputs contain only page 1")
        try:
            with Image.open(path) as original:
                orientation = original.getexif().get(274, 1)
                resolution = original.info.get("dpi")
                image = ImageOps.exif_transpose(original)
                if "A" in image.getbands() or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)
                pixels = np.asarray(image.convert("L" if grayscale else "RGB")).copy()
                meta["exif_orientation_applied"] = int(orientation)
                if resolution is not None:
                    resolution = np.asarray(resolution, dtype=float)
                    if resolution.shape == (2,) and np.isfinite(resolution).all() and np.all(resolution > 0):
                        if orientation in (5, 6, 7, 8):
                            resolution = resolution[::-1]
                        meta["dpi_xy"] = resolution.tolist()
                        meta["captured_size_mm"] = (np.array([pixels.shape[1], pixels.shape[0]]) * 25.4 / resolution).tolist()
        except Exception as exc:
            raise ValueError(f"could not decode image {path}: {exc}") from exc
    meta["source_size_px"] = [pixels.shape[1], pixels.shape[0]]
    if not grayscale:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    return DocumentPage(pixels, meta)


def load_document_page_info(
    input_path: str | Path,
    *,
    page_index: int = 0,
    dpi: float = DEFAULT_DPI,
    grayscale: bool = False,
) -> LoadedDocumentPage:
    """Load pixels together with the geometry used by input normalization.

    This compatibility view is derived from :func:`load_document` so scanner
    normalization and physical registration share one decoder and one set of
    PDF/EXIF rules.
    """
    document = load_document(
        input_path,
        page_index=page_index,
        dpi=dpi,
        grayscale=grayscale,
    )
    height_px, width_px = document.image.shape[:2]
    is_pdf = document.metadata.get("kind") == "pdf"
    physical_value = document.metadata.get(
        "page_size_mm" if is_pdf else "captured_size_mm"
    )
    physical = (
        None
        if physical_value is None
        else (float(physical_value[0]), float(physical_value[1]))
    )
    dpi_value = document.metadata.get("dpi_xy")
    source_dpi = (
        (float(dpi), float(dpi))
        if is_pdf
        else (
            None
            if dpi_value is None
            else (float(dpi_value[0]), float(dpi_value[1]))
        )
    )
    rotation = document.metadata.get("rotation_degrees") if is_pdf else None
    return LoadedDocumentPage(
        pixels=document.image,
        pixel_dimensions=(width_px, height_px),
        physical_dimensions_mm=physical,
        source_dpi=source_dpi,
        pdf_rotation_degrees=None if rotation is None else int(rotation),
        is_pdf=is_pdf,
    )


def load_document_page(input_path: str | Path, *, page_index: int = 0,
                       dpi: float = DEFAULT_DPI, grayscale: bool = False) -> np.ndarray:
    """Backwards-compatible pixel-only API; use load_document to retain geometry."""
    return load_document(input_path, page_index=page_index, dpi=dpi, grayscale=grayscale).image
