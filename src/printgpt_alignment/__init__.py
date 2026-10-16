"""Physical A4 registration and alignment tools for PrintGPT."""

from .detection import (
    BoundaryThresholds,
    DetectionResult,
    QualityThresholds,
    detect_and_rectify,
    detect_and_rectify_document,
)
from .document import LoadedDocumentPage, load_document_page, load_document_page_info
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_px, px_to_mm

__all__ = [
    "A4CalibrationLayout",
    "BoundaryThresholds",
    "DEFAULT_LAYOUT",
    "DetectionResult",
    "LoadedDocumentPage",
    "QualityThresholds",
    "detect_and_rectify",
    "detect_and_rectify_document",
    "load_document_page",
    "load_document_page_info",
    "mm_to_px",
    "px_to_mm",
]
