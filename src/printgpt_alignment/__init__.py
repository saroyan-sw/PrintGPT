"""Physical A4 registration and alignment tools for PrintGPT."""

from .detection import DetectionResult, QualityThresholds, detect_and_rectify
from .document import load_document_page
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_px, px_to_mm

__all__ = [
    "A4CalibrationLayout",
    "DEFAULT_LAYOUT",
    "DetectionResult",
    "QualityThresholds",
    "detect_and_rectify",
    "load_document_page",
    "mm_to_px",
    "px_to_mm",
]
