"""Marker and marker-free A4 detection, quality assessment, and rectification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import atan2, degrees, exp
from typing import Any, Literal

import cv2
import numpy as np

from .calibration_page import aruco_dictionary
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT

InputKind = Literal["auto", "scan", "photo"]


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    min_marker_side_px: float = 10.0
    min_page_area_ratio: float = 0.04
    max_page_area_ratio: float = 1.80
    max_marker_area_ratio: float = 10.0
    min_inlier_count: int = 12
    max_reprojection_rmse_px: float = 8.5
    max_reprojection_error_px: float = 12.0
    min_confidence: float = 0.60


@dataclass(frozen=True, slots=True)
class BoundaryThresholds:
    """Conservative acceptance thresholds for marker-free document scans."""

    min_short_side_px: int = 300
    min_page_area_ratio: float = 0.20
    min_page_background_contrast: float = 10.0
    max_a4_aspect_error: float = 0.12
    full_frame_aspect_error: float = 0.035
    min_scan_paper_fraction: float = 0.70
    min_scan_border_paper_fraction: float = 0.50
    min_laplacian_variance: float = 12.0
    min_tonal_range: float = 18.0
    min_edge_fraction: float = 0.0005
    max_detection_dimension: int = 1600
    min_confidence: float = 0.60


@dataclass(slots=True)
class DetectionResult:
    accepted: bool
    reason: str
    confidence: float
    detected_ids: tuple[int, ...]
    orientation_degrees: int | None = None
    source_to_canonical: np.ndarray | None = None
    canonical_to_source: np.ndarray | None = None
    warped: np.ndarray | None = None
    source_marker_corners: dict[int, np.ndarray] | None = None
    reprojection_rmse_px: float | None = None
    reprojection_max_px: float | None = None
    reprojection_inlier_max_px: float | None = None
    inlier_count: int = 0
    page_area_ratio: float | None = None
    marker_area_ratio: float | None = None
    alignment_method: str = "aruco"
    source_page_corners: np.ndarray | None = None
    blur_variance: float | None = None
    tonal_range: float | None = None
    edge_fraction: float | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("warped", None)
        if self.source_to_canonical is not None:
            result["source_to_canonical"] = self.source_to_canonical.tolist()
        if self.canonical_to_source is not None:
            result["canonical_to_source"] = self.canonical_to_source.tolist()
        if self.source_marker_corners is not None:
            result["source_marker_corners"] = {
                str(marker_id): corners.tolist()
                for marker_id, corners in self.source_marker_corners.items()
            }
        if self.source_page_corners is not None:
            result["source_page_corners"] = self.source_page_corners.tolist()
        return result


def _reject(reason: str, detected_ids: tuple[int, ...], **kwargs: Any) -> DetectionResult:
    return DetectionResult(False, reason, 0.0, detected_ids, **kwargs)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("input image is empty")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


def _transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points.astype(np.float32)[None, :, :], homography)[0]


def _canonical_marker_corners(layout: A4CalibrationLayout, marker_id: int) -> np.ndarray:
    return np.asarray(layout.marker_by_id(marker_id).corners_px(layout.dpi), dtype=np.float32)


def _associate_corners(
    source_corners: np.ndarray,
    marker_id: int,
    initial_source_to_canonical: np.ndarray,
    layout: A4CalibrationLayout,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = _transform(source_corners, initial_source_to_canonical)
    expected = _canonical_marker_corners(layout, marker_id)
    best_order: tuple[int, ...] | None = None
    best_cost = float("inf")
    for order in permutations(range(4)):
        cost = float(np.sum((predicted[list(order)] - expected) ** 2))
        if cost < best_cost:
            best_cost = cost
            best_order = order
    assert best_order is not None
    return source_corners[list(best_order)], expected


def _orientation(marker_centers: dict[int, np.ndarray]) -> int:
    vector = marker_centers[11] - marker_centers[10]
    angle = degrees(atan2(float(vector[1]), float(vector[0])))
    return int((round(angle / 90.0) * 90) % 360)


def detect_and_rectify(
    image: np.ndarray,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    thresholds: QualityThresholds = QualityThresholds(),
) -> DetectionResult:
    """Detect the expected marker quartet and warp an accepted image to A4."""
    gray = _to_gray(image)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(layout), parameters)
    detected_corners, detected_ids_array, _ = detector.detectMarkers(gray)
    if detected_ids_array is None:
        return _reject("no ArUco markers detected", ())

    ids = [int(value) for value in detected_ids_array.ravel()]
    detected_ids = tuple(sorted(ids))
    relevant: dict[int, list[np.ndarray]] = {}
    expected = set(layout.expected_marker_ids)
    for marker_id, corners in zip(ids, detected_corners, strict=True):
        if marker_id in expected:
            relevant.setdefault(marker_id, []).append(corners.reshape(4, 2).astype(np.float32))

    missing = sorted(expected - relevant.keys())
    duplicates = sorted(marker_id for marker_id, values in relevant.items() if len(values) != 1)
    if missing:
        return _reject(f"missing expected markers: {missing}", detected_ids)
    if duplicates:
        return _reject(f"duplicate expected markers: {duplicates}", detected_ids)

    source_by_id = {marker_id: values[0] for marker_id, values in relevant.items()}
    side_lengths = []
    marker_areas = []
    for corners in source_by_id.values():
        side_lengths.extend(
            float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
            for index in range(4)
        )
        marker_areas.append(abs(float(cv2.contourArea(corners))))
    if min(side_lengths) < thresholds.min_marker_side_px:
        return _reject(
            f"marker side below {thresholds.min_marker_side_px:.1f} px",
            detected_ids,
            source_marker_corners=source_by_id,
        )
    marker_area_ratio = max(marker_areas) / max(min(marker_areas), 1e-9)
    if marker_area_ratio > thresholds.max_marker_area_ratio:
        return _reject(
            f"marker scale ratio {marker_area_ratio:.2f} is implausible",
            detected_ids,
            source_marker_corners=source_by_id,
            marker_area_ratio=marker_area_ratio,
        )

    marker_centers = {
        marker_id: corners.mean(axis=0) for marker_id, corners in source_by_id.items()
    }
    source_centers = np.asarray(
        [marker_centers[marker.marker_id] for marker in layout.markers], dtype=np.float32
    )
    canonical_centers = np.asarray(
        [marker.center_mm.to_pixels(layout.dpi) for marker in layout.markers], dtype=np.float32
    )
    initial = cv2.getPerspectiveTransform(source_centers, canonical_centers)
    if not np.isfinite(initial).all() or abs(float(np.linalg.det(initial))) < 1e-12:
        return _reject("degenerate initial homography", detected_ids, source_marker_corners=source_by_id)

    source_points: list[np.ndarray] = []
    canonical_points: list[np.ndarray] = []
    for marker in layout.markers:
        ordered_source, expected_corners = _associate_corners(
            source_by_id[marker.marker_id], marker.marker_id, initial, layout
        )
        source_points.append(ordered_source)
        canonical_points.append(expected_corners)
    source_array = np.concatenate(source_points).astype(np.float32)
    canonical_array = np.concatenate(canonical_points).astype(np.float32)
    homography, inlier_mask = cv2.findHomography(
        source_array,
        canonical_array,
        method=cv2.RANSAC,
        ransacReprojThreshold=thresholds.max_reprojection_rmse_px,
    )
    if homography is None or inlier_mask is None or not np.isfinite(homography).all():
        return _reject("homography estimation failed", detected_ids, source_marker_corners=source_by_id)
    inlier_count = int(inlier_mask.ravel().sum())
    if inlier_count < thresholds.min_inlier_count:
        return _reject(
            f"only {inlier_count}/16 marker corners support the homography",
            detected_ids,
            source_marker_corners=source_by_id,
            source_to_canonical=homography,
            inlier_count=inlier_count,
        )

    projected = _transform(source_array, homography)
    errors = np.linalg.norm(projected - canonical_array, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    inlier_errors = errors[inlier_mask.ravel().astype(bool)]
    inlier_max_error = float(np.max(inlier_errors))
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return _reject("homography is singular", detected_ids, source_marker_corners=source_by_id)

    page_source = _transform(np.asarray(layout.page_corners_px, dtype=np.float32), inverse)
    if not cv2.isContourConvex(page_source.astype(np.float32)):
        return _reject("recovered page boundary is not convex", detected_ids, source_marker_corners=source_by_id)
    image_area = float(gray.shape[0] * gray.shape[1])
    page_area_ratio = abs(float(cv2.contourArea(page_source))) / image_area
    if not thresholds.min_page_area_ratio <= page_area_ratio <= thresholds.max_page_area_ratio:
        return _reject(
            f"page area ratio {page_area_ratio:.3f} is outside accepted range",
            detected_ids,
            source_marker_corners=source_by_id,
            page_area_ratio=page_area_ratio,
        )

    inlier_score = inlier_count / 16.0
    residual_score = exp(-rmse / max(thresholds.max_reprojection_rmse_px, 1e-6))
    maximum_score = exp(-inlier_max_error / max(thresholds.max_reprojection_error_px, 1e-6))
    scale_score = max(0.0, 1.0 - (marker_area_ratio - 1.0) / max(thresholds.max_marker_area_ratio - 1.0, 1e-6))
    size_score = min(1.0, min(side_lengths) / (2.0 * thresholds.min_marker_side_px))
    confidence = float(
        0.30 * residual_score
        + 0.20 * maximum_score
        + 0.25 * inlier_score
        + 0.15 * scale_score
        + 0.10 * size_score
    )

    common = dict(
        orientation_degrees=_orientation(marker_centers),
        source_to_canonical=homography,
        canonical_to_source=inverse,
        source_marker_corners=source_by_id,
        reprojection_rmse_px=rmse,
        reprojection_max_px=max_error,
        reprojection_inlier_max_px=inlier_max_error,
        inlier_count=inlier_count,
        page_area_ratio=page_area_ratio,
        marker_area_ratio=marker_area_ratio,
    )
    if rmse > thresholds.max_reprojection_rmse_px:
        return DetectionResult(False, f"reprojection RMSE {rmse:.2f} px is too high", confidence, detected_ids, **common)
    if inlier_max_error > thresholds.max_reprojection_error_px:
        return DetectionResult(False, f"maximum inlier reprojection error {inlier_max_error:.2f} px is too high", confidence, detected_ids, **common)
    if confidence < thresholds.min_confidence:
        return DetectionResult(False, f"quality score {confidence:.3f} is too low", confidence, detected_ids, **common)

    border_value: int | tuple[int, int, int]
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    warped = cv2.warpPerspective(
        image,
        homography,
        layout.raster_size_px,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return DetectionResult(True, "accepted", confidence, detected_ids, warped=warped, **common)


_RESCAN_GUIDANCE = (
    "Please scan again with the entire page visible, good lighting, sharp focus, "
    "and a contrasting background."
)


def _marker_free_reject(reason: str, marker_result: DetectionResult, **kwargs: Any) -> DetectionResult:
    return DetectionResult(
        False,
        f"Could not reliably detect the document: {reason}. {_RESCAN_GUIDANCE}",
        0.0,
        marker_result.detected_ids,
        alignment_method="page_boundary",
        **kwargs,
    )


def _order_quad(corners: np.ndarray) -> np.ndarray:
    """Return quadrilateral corners in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()
    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = points[int(np.argmin(sums))]
    ordered[2] = points[int(np.argmax(sums))]
    ordered[1] = points[int(np.argmin(differences))]
    ordered[3] = points[int(np.argmax(differences))]
    if len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("page corners are ambiguous")
    return ordered


def _quad_dimensions(corners: np.ndarray) -> tuple[float, float]:
    top = float(np.linalg.norm(corners[1] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    return (top + bottom) / 2.0, (left + right) / 2.0


def _portrait_order(corners: np.ndarray) -> tuple[np.ndarray, int]:
    width, height = _quad_dimensions(corners)
    if width <= height:
        return corners, 0
    # A marker-free page has no semantic cue for clockwise versus counter-clockwise.
    # Choose a deterministic counter-clockwise correction for landscape captures.
    return corners[[1, 2, 3, 0]], 90


def _aspect_error(corners: np.ndarray, layout: A4CalibrationLayout) -> float:
    width, height = _quad_dimensions(corners)
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side <= 0:
        return float("inf")
    observed = short_side / long_side
    expected = min(layout.page_width_mm, layout.page_height_mm) / max(
        layout.page_width_mm, layout.page_height_mm
    )
    return abs(observed - expected) / expected


def _scan_quality(gray: np.ndarray) -> tuple[float, float, float]:
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low, high = np.percentile(gray, (0.01, 99.99))
    tonal_range = float(high - low)
    edges = cv2.Canny(gray, 50, 150)
    edge_fraction = float(np.count_nonzero(edges) / edges.size)
    return blur_variance, tonal_range, edge_fraction


def _scaled_gray(gray: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    largest = max(gray.shape)
    if largest <= max_dimension:
        return gray, 1.0
    scale = max_dimension / float(largest)
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _full_frame_quad(
    gray: np.ndarray,
    layout: A4CalibrationLayout,
    thresholds: BoundaryThresholds,
) -> np.ndarray | None:
    height, width = gray.shape
    corners = np.asarray(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float32,
    )
    # A scanner export already uses the image bounds as the page boundary.
    # Its aspect ratio is not evidence of scan quality or physical paper size.
    # Require light paper across the image and on every side, allowing narrow
    # scanner bars/speckles. A dark surround belongs to the photograph path.
    paper = gray >= 180
    band = max(1, int(round(min(height, width) * 0.08)))
    borders = (paper[:band], paper[-band:], paper[:, :band], paper[:, -band:])
    if float(paper.mean()) < thresholds.min_scan_paper_fraction:
        return None
    if any(float(border.mean()) < thresholds.min_scan_border_paper_fraction for border in borders):
        return None
    return corners


def _find_page_quad(
    gray: np.ndarray,
    layout: A4CalibrationLayout,
    thresholds: BoundaryThresholds,
) -> tuple[np.ndarray | None, float | None, float | None]:
    small, scale = _scaled_gray(gray, thresholds.max_detection_dimension)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    median = float(np.median(blurred))
    lower = int(max(0.0, 0.66 * median))
    upper = int(min(255.0, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    candidates: list[tuple[float, float, float, np.ndarray]] = []
    for contour in contours:
        area = abs(float(cv2.contourArea(contour)))
        area_ratio = area / image_area
        if area_ratio < thresholds.min_page_area_ratio:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue
        try:
            ordered = _order_quad(approximation.reshape(4, 2))
        except ValueError:
            continue
        # A clipped photograph must not be completed with invented corners.
        if (ordered[:, 0].min() <= 2 or ordered[:, 1].min() <= 2
                or ordered[:, 0].max() >= small.shape[1] - 3
                or ordered[:, 1].max() >= small.shape[0] - 3):
            continue
        aspect_error = _aspect_error(ordered, layout)
        if aspect_error > thresholds.max_a4_aspect_error:
            continue
        page_mask = np.zeros(small.shape, dtype=np.uint8)
        cv2.fillConvexPoly(page_mask, np.rint(ordered).astype(np.int32), 255)
        inside = blurred[page_mask == 255]
        outside = blurred[page_mask == 0]
        if outside.size == 0:
            continue
        page_background_contrast = float(np.median(inside) - np.median(outside))
        if page_background_contrast < thresholds.min_page_background_contrast:
            continue
        score = area_ratio - aspect_error
        candidates.append((score, area_ratio, aspect_error, ordered / scale))
    if not candidates:
        return None, None, None
    _, area_ratio, aspect_error, corners = max(candidates, key=lambda item: item[0])
    return corners.astype(np.float32), area_ratio, aspect_error


def detect_and_rectify_document(
    image: np.ndarray,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    marker_thresholds: QualityThresholds = QualityThresholds(),
    boundary_thresholds: BoundaryThresholds = BoundaryThresholds(),
    *,
    input_kind: InputKind = "auto",
) -> DetectionResult:
    """Use markers, an already scanned page, or a photographed page boundary.

    Scan mode treats the complete image as the page. Auto mode infers that case
    from paper reaching all four sides, never from the image's aspect ratio.
    Photos require visible page edges. Non-A4 scans are fitted without stretching;
    their physical dimensions cannot be recovered from pixels alone.
    """
    if input_kind not in ("auto", "scan", "photo"):
        raise ValueError("input_kind must be auto, scan, or photo")
    marker_result = detect_and_rectify(image, layout, marker_thresholds)
    if marker_result.accepted:
        return marker_result

    gray = _to_gray(image)
    if min(gray.shape) < boundary_thresholds.min_short_side_px:
        return _marker_free_reject(
            f"image is too small ({gray.shape[1]} x {gray.shape[0]} px)", marker_result
        )

    quality_gray, _ = _scaled_gray(gray, boundary_thresholds.max_detection_dimension)
    blur_variance, tonal_range, edge_fraction = _scan_quality(quality_gray)
    quality = {
        "blur_variance": blur_variance,
        "tonal_range": tonal_range,
        "edge_fraction": edge_fraction,
    }
    if blur_variance < boundary_thresholds.min_laplacian_variance:
        return _marker_free_reject("the scan is too blurry", marker_result, **quality)
    if tonal_range < boundary_thresholds.min_tonal_range:
        return _marker_free_reject("the scan has too little contrast", marker_result, **quality)
    if edge_fraction < boundary_thresholds.min_edge_fraction:
        return _marker_free_reject("the document detail is too faint", marker_result, **quality)

    page_corners = None
    if input_kind == "scan":
        height, width = gray.shape
        page_corners = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    elif input_kind == "auto":
        page_corners = _full_frame_quad(gray, layout, boundary_thresholds)
    is_scan = page_corners is not None
    page_area_ratio = 1.0 if page_corners is not None else None
    aspect_error = _aspect_error(page_corners, layout) if page_corners is not None else None
    if page_corners is None:
        page_corners, page_area_ratio, aspect_error = _find_page_quad(
            gray, layout, boundary_thresholds
        )
    if page_corners is None or page_area_ratio is None or aspect_error is None:
        return _marker_free_reject(
            "the photograph does not show a clear, complete page boundary; "
            "if this is an already cropped scanner export, choose Scanned page", marker_result, **quality
        )

    source, orientation = _portrait_order(page_corners)
    destination = np.asarray(layout.page_corners_px, dtype=np.float32)
    warnings = ()
    if is_scan:
        # Preserve source proportions. Stretching arbitrary scanner exports to
        # 210 x 297 mm silently distorts ink and gives false physical precision.
        width, height = _quad_dimensions(source)
        scale = min(layout.raster_width_px / width, layout.raster_height_px / height)
        fitted_width, fitted_height = width * scale, height * scale
        x = (layout.raster_width_px - fitted_width) / 2
        y = (layout.raster_height_px - fitted_height) / 2
        destination = np.asarray(
            [[x, y], [x + fitted_width, y], [x + fitted_width, y + fitted_height], [x, y + fitted_height]],
            dtype=np.float32,
        )
        if aspect_error > boundary_thresholds.full_frame_aspect_error:
            warnings = (
                "This scan has different proportions from A4. It was fitted to A4 without stretching. "
                "Its original physical size is unknown, so an overlay may not line up on the original sheet.",
            )
    homography = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    if not np.isfinite(homography).all() or abs(float(np.linalg.det(homography))) < 1e-12:
        return _marker_free_reject("the page perspective is ambiguous", marker_result, **quality)
    try:
        inverse = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return _marker_free_reject("the page perspective is singular", marker_result, **quality)

    aspect_score = 1.0 if is_scan else max(0.0, 1.0 - aspect_error / boundary_thresholds.max_a4_aspect_error)
    area_score = min(1.0, page_area_ratio / 0.65)
    sharpness_score = min(1.0, blur_variance / (4.0 * boundary_thresholds.min_laplacian_variance))
    contrast_score = min(1.0, tonal_range / (4.0 * boundary_thresholds.min_tonal_range))
    confidence = float(
        0.35 * aspect_score + 0.30 * area_score + 0.20 * sharpness_score + 0.15 * contrast_score
    )
    if confidence < boundary_thresholds.min_confidence:
        return _marker_free_reject(
            f"page-boundary confidence {confidence:.3f} is too low", marker_result, **quality
        )

    border_value: int | tuple[int, int, int]
    border_value = 255 if image.ndim == 2 else (255, 255, 255)
    warped = cv2.warpPerspective(
        image,
        homography,
        layout.raster_size_px,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return DetectionResult(
        True,
        "accepted using the scanned image bounds" if is_scan else "accepted using the detected page boundary",
        confidence,
        marker_result.detected_ids,
        alignment_method="scan_frame" if is_scan else "page_boundary",
        orientation_degrees=orientation,
        source_to_canonical=homography,
        canonical_to_source=inverse,
        warped=warped,
        page_area_ratio=page_area_ratio,
        source_page_corners=page_corners,
        blur_variance=blur_variance,
        tonal_range=tonal_range,
        edge_fraction=edge_fraction,
        warnings=warnings,
    )
