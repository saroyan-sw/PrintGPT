"""Integrated offline PrintGPT MVP with an explicit approval/export gate."""

from .input_normalization import (
    AppliedTransform,
    InputMode,
    InputNormalizationResult,
    OrientationChoice,
    ScannerA4Tolerance,
    normalize_document_input,
)

from .pipeline import (
    PRINT_AT_ACTUAL_SIZE,
    ApprovedExport,
    CandidateView,
    MvpRun,
    approve_and_export,
    canonicalize_document,
    rectangular_allowed_region,
    run_offline_document,
)

__all__ = [
    "PRINT_AT_ACTUAL_SIZE",
    "AppliedTransform",
    "ApprovedExport",
    "CandidateView",
    "InputMode",
    "InputNormalizationResult",
    "MvpRun",
    "OrientationChoice",
    "ScannerA4Tolerance",
    "approve_and_export",
    "canonicalize_document",
    "normalize_document_input",
    "rectangular_allowed_region",
    "run_offline_document",
]
