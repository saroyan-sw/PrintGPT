"""ArUco-based A4 detection, quality assessment, and rectification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from math import atan2, degrees, exp
from typing import Any

import cv2
import numpy as np

from .calibration_page import aruco_dictionary
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT


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
