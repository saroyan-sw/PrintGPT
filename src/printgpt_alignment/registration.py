"""Evidence-based source-to-paper registration, separate from preview acceptance.

The canonical raster spans exactly 210 x 297 mm at its outer pixel edges.
No inferred white border or aspect ratio is sufficient evidence of paper origin.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np

from .document import DocumentPage
from .detection import detect_and_rectify_document
from .geometry import DEFAULT_LAYOUT, A4CalibrationLayout


@dataclass(frozen=True)
class RegistrationOptions:
    mode: Literal["auto", "full_page", "corners", "metadata", "crop"] = "auto"
    confirmed: bool = False
    # Corners in physical TL, TR, BR, BL order, in EXIF-normalized source pixels.
    corners_px: tuple[tuple[float, float], ...] | None = None
    dpi_xy: tuple[float, float] | None = None
    origin_mm: tuple[float, float] | None = None
    page_size_mm: tuple[float, float] = (210.0, 297.0)
    allow_nonuniform_resize: bool = False
    control_points_px: tuple[tuple[float, float], ...] | None = None
    control_points_mm: tuple[tuple[float, float], ...] | None = None
    measurement_uncertainty_mm: float | None = None


def _pair(value, label, *, positive=False):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (2,) or not np.isfinite(arr).all() or (positive and np.any(arr <= 0)):
        raise ValueError(f"{label} must contain two finite {'positive ' if positive else ''}numbers")
    return arr


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return cv2.perspectiveTransform(points.reshape(1, -1, 2), np.asarray(transform, dtype=float))[0]


def register_document(page: DocumentPage, *, layout: A4CalibrationLayout = DEFAULT_LAYOUT,
                      input_kind="auto", options: RegistrationOptions | None = None):
    """Return warped BGR/gray pixels and a JSON-safe evidence report.

    User confirmation means geometry was checked externally; it is not a measured
    accuracy certificate. ArUco remains automatic under the known printed-layout
    assumption. Printer feed accuracy is NEVER inferred from a successful scan.
    """
    options = options or RegistrationOptions()
    if not isinstance(options.confirmed, bool) or not isinstance(options.allow_nonuniform_resize, bool):
        raise ValueError("confirmation and resize acknowledgement must be booleans")
    if options.mode not in ("auto", "full_page", "corners", "metadata", "crop"):
        raise ValueError("unknown registration mode")
    size = _pair(options.page_size_mm, "page_size_mm", positive=True)
    image, meta = page.image, page.metadata
    h, w = image.shape[:2]
    bounds = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    paper_corners = np.array([[0, 0], [size[0], 0], size, [0, size[1]]], dtype=np.float32)
    mm_to_canonical = np.diag([layout.raster_width_px / layout.page_width_mm,
                               layout.raster_height_px / layout.page_height_mm, 1.])
    detection = detect_and_rectify_document(image, layout, input_kind=input_kind)
    report = detection.to_dict()
    warnings = list(report.get("warnings", ()))
    registered = detection.accepted and detection.alignment_method == "aruco" and options.mode == "auto"
    evidence = "known ArUco layout; assumes calibration was printed at actual size" if registered else "unknown scale and/or paper origin"
    H = detection.source_to_canonical
    source_corners = bounds
    if detection.source_page_corners is not None:
        source_corners = np.asarray(detection.source_page_corners, dtype=np.float32)
    target_mm = None
    if options.mode == "full_page":
        # Explicit whole-sheet assertion; not a content bounding-box heuristic.
        aspect_error = abs((w / h) / (size[0] / size[1]) - 1)
        if aspect_error > .02 and not options.allow_nonuniform_resize:
            raise ValueError("image aspect ratio differs from the paper: use paper corners or measured crop geometry; only allow nonuniform resize if the complete sheet was resized")
        source_corners, target_mm = bounds, paper_corners
        evidence = "user-confirmed full paper bounds and physical size"
    elif options.mode == "corners":
        source_corners = np.asarray(options.corners_px, dtype=np.float32)
        if source_corners.shape != (4, 2) or not np.isfinite(source_corners).all():
            raise ValueError("provide four finite paper corners in physical TL, TR, BR, BL order")
        if np.any(source_corners < 0) or np.any(source_corners > [w, h]):
            raise ValueError("paper corners must be visible within the source; use measured crop offsets for missing edges")
        if not cv2.isContourConvex(source_corners) or cv2.contourArea(source_corners, oriented=True) <= 100:
            raise ValueError("paper corners must be a nondegenerate clockwise quadrilateral (TL, TR, BR, BL)")
        target_mm = paper_corners
        evidence = "user-confirmed physical paper corners and size"
    elif options.mode == "crop":
        resolution = _pair(options.dpi_xy or meta.get("dpi_xy"), "dpi_xy", positive=True)
        origin = _pair(options.origin_mm, "origin_mm")
        source_corners, target_mm = bounds, bounds * (25.4 / resolution) + origin
        evidence = "user-confirmed scan DPI and physical origin; assumes no skew/perspective or subsequent resizing"
    elif options.mode == "metadata" or (options.mode == "auto" and not registered and meta.get("source_corners_mm")):
        if meta.get("kind") == "pdf" and meta.get("source_corners_mm"):
            size = _pair(meta["page_size_mm"], "PDF page size", positive=True)
            source_corners, target_mm = bounds, np.asarray(meta["source_corners_mm"], dtype=np.float32)
            evidence = "PDF MediaBox/CropBox/UserUnit/rotation; original scan placement requires confirmation"
        elif options.mode == "metadata" and meta.get("dpi_xy"):
            resolution = _pair(meta["dpi_xy"], "metadata dpi", positive=True)
            captured_size = np.array([w, h]) * 25.4 / resolution
            if np.max(abs(captured_size - size)) > .5:
                raise ValueError("DPI dimensions disagree with a complete sheet by more than 0.5 mm; use measured crop offsets or paper corners")
            source_corners, target_mm = bounds, bounds * (25.4 / resolution)
            evidence = "scanner DPI plus user-confirmed uncropped, unskewed paper origin"
        elif options.mode == "metadata":
            raise ValueError("no usable physical metadata; confirm full-page bounds, paper corners, or measured crop geometry")
    if target_mm is not None:
        H = mm_to_canonical @ cv2.getPerspectiveTransform(source_corners.astype(np.float32), target_mm.astype(np.float32))
        registered = bool(options.confirmed and options.mode != "auto" and np.allclose(size, [210, 297], atol=.05, rtol=0))
        report.update(alignment_method="physical_" + options.mode, source_page_corners=source_corners.tolist())
        # Geometry is explicit: blank forms do not need text/contrast heuristics.
        report.update(accepted=True, reason="explicit file/paper geometry used", confidence=None)
    elif not detection.accepted or H is None:
        raise ValueError(f"page alignment rejected: {detection.reason}")
    registered = bool(registered and np.allclose(size, [210, 297], atol=.05, rtol=0))
    # Whole-page Laplacian variance confuses sparse sharp marks with blur. Check
    # local edge strength too before blocking an explicitly mapped sparse page.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if max(gray.shape) > 1600:
        gray = cv2.resize(gray, None, fx=1600/max(gray.shape), fy=1600/max(gray.shape), interpolation=cv2.INTER_AREA)
    local_edge_strength = float(np.percentile(np.abs(cv2.Laplacian(gray, cv2.CV_64F)), 99.99))
    if (detection.blur_variance is not None and detection.blur_variance < 12
            and detection.tonal_range is not None and detection.tonal_range > 18
            and local_edge_strength < 25):
        registered = False
        warnings.append("Insufficient sharp detail for reliable registration/protection. Obtain a sharper capture before printing; geometry confirmation cannot recover missing ink detail.")
    if not registered:
        warnings.append("Preview only: physical scale and paper origin are not confirmed. Do not print this placement onto the original sheet.")
    if meta.get("geometry_warning"):
        warnings.append(meta["geometry_warning"])
    if registered and options.mode == "auto":
        warnings.append("ArUco registration assumes actual-size markers. Reprojection residual is not a printer-accuracy measurement.")
    if not np.allclose(size, [210, 297], atol=.05, rtol=0):
        warnings.append("Physical page is not portrait A4; export onto an A4 original is blocked.")
    H = np.asarray(H, dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("invalid or singular registration transform")
    # Registration cannot recover detail beyond the captured sampling scale.
    probes = np.vstack([source_corners, source_corners.mean(axis=0)])
    source_to_mm = np.linalg.inv(mm_to_canonical) @ H
    origin_mm = transform_points(probes, source_to_mm)
    dx = transform_points(probes + [1,0], source_to_mm) - origin_mm
    dy = transform_points(probes + [0,1], source_to_mm) - origin_mm
    # Maximum diagonal also accommodates anisotropic/skewed sampling cells.
    sampling_mm = float(np.maximum(np.linalg.norm(dx+dy,axis=1),np.linalg.norm(dx-dy,axis=1)).max())
    if not np.isfinite(sampling_mm) or sampling_mm >= .5:
        registered = False
        warnings.append("Capture sampling is too coarse for the 0.5 mm registration target; use a higher-resolution original capture before exporting.")
    warped = cv2.warpPerspective(image, H, (layout.raster_width_px, layout.raster_height_px), borderValue=(255,255,255))
    observed = transform_points(source_corners, H)
    control_report = None
    source_bound = None
    if options.control_points_px is not None or options.control_points_mm is not None:
        controls = np.asarray(options.control_points_mm, dtype=float)
        uncertainty = options.measurement_uncertainty_mm
        if controls.ndim != 2 or controls.shape[1] != 2 or len(controls) < 9 or not np.isfinite(controls).all() or np.any(np.ptp(controls, axis=0) < [125,175]):
            raise ValueError("use at least nine independent control points spread across the paper")
        if uncertainty is None or not np.isfinite(uncertainty) or uncertainty <= 0:
            raise ValueError("control-point measurements need positive measurement_uncertainty_mm")
        source_controls = np.asarray(options.control_points_px, dtype=float)
        if source_controls.shape != controls.shape or not np.isfinite(source_controls).all():
            raise ValueError("control points must be matching finite Nx2 arrays")
        fit_points = source_corners
        if registered and options.mode == "auto":
            marker_points = list(detection.source_marker_corners.values())
            if marker_points:
                fit_points = np.concatenate(marker_points).reshape(-1,2)
        if (len(set(map(tuple, controls))) != len(controls)
                or np.min(np.linalg.norm(source_controls[:,None] - fit_points[None,:], axis=2)) < 1.):
            raise ValueError("control points must be independent distinct locations, not registration corners")
        control_report = evaluate_control_points(options.control_points_px, controls, H, layout=layout)
        source_bound = control_report["max_error_mm"] + float(uncertainty)
        if source_bound >= .5:
            registered = False
            warnings.append("Independent source checks exceed the 0.5 mm target including measurement uncertainty; correct geometry before export.")
    report.update(
        source_to_canonical=H.tolist(), canonical_to_source=np.linalg.inv(H).tolist(),
        observed_polygon_canonical_px=observed.tolist(),
        warnings=warnings, source_metadata=meta,
        local_edge_strength=local_edge_strength,
        registration={"status": "registered" if registered else "preview_only", "export_allowed": registered,
                      "evidence": evidence, "options": asdict(options), "page_size_mm": size.tolist(),
                      "scale_and_origin": "established under recorded assumptions" if registered else "inferred or unknown",
                      "independent_controls": control_report,
                      "max_sampled_pixel_diagonal_mm": sampling_mm,
                      "measured_source_accuracy_mm": source_bound, "printer_accuracy": "unmeasured"},
        coordinate_system={"origin": "physical paper top-left", "axes": "x right, y down",
                           "convention": "outer pixel edges; raster spans exact page dimensions",
                           "canonical_px_to_mm": np.linalg.inv(mm_to_canonical).tolist(),
                           "source_to_mm": (np.linalg.inv(mm_to_canonical) @ H).tolist(),
                           "mm_to_pdf_pt": [[72/25.4,0,0],[0,-72/25.4,297*72/25.4],[0,0,1]]},
    )
    return warped, report


def evaluate_control_points(source_points_px, expected_mm, source_to_canonical, *, layout=DEFAULT_LAYOUT):
    """Independent points only: do not reuse corners/markers used to fit H."""
    points = np.asarray(source_points_px, dtype=float)
    expected = np.asarray(expected_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape != expected.shape or len(points) < 1 or not np.isfinite(np.r_[points, expected]).all():
        raise ValueError("control points must be matching nonempty finite Nx2 arrays")
    recovered_px = transform_points(points, source_to_canonical)
    scale = np.array([layout.page_width_mm/layout.raster_width_px, layout.page_height_mm/layout.raster_height_px])
    recovered = recovered_px * scale
    errors = np.linalg.norm(recovered - expected, axis=1)
    return {"mean_error_mm": float(errors.mean()), "max_error_mm": float(errors.max()),
            "errors_mm": errors.tolist(), "recovered_mm": recovered.tolist(),
            "expected_mm": expected.tolist(), "meets_0_5mm_target": bool(errors.max() < .5)}
