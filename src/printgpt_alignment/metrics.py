"""Independent alignment metrics and expected-versus-recovered overlays."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT, px_to_mm


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    corner_errors_px: tuple[float, float, float, float]
    corner_errors_mm: tuple[float, float, float, float]
    mean_error_px: float
    mean_error_mm: float
    rmse_px: float
    rmse_mm: float
    p95_error_px: float
    p95_error_mm: float
    max_error_px: float
    max_error_mm: float
    evaluation_point_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    return cv2.perspectiveTransform(points[None, :, :], homography)[0]


def evaluation_grid(
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    columns: int = 9,
    rows: int = 13,
) -> np.ndarray:
    """Return an independent page-wide grid that includes all four corners."""
    if columns < 2 or rows < 2:
        raise ValueError("evaluation grid needs at least two rows and columns")
    xs = np.linspace(0.0, float(layout.raster_width_px), columns)
    ys = np.linspace(0.0, float(layout.raster_height_px), rows)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)


def evaluate_alignment(
    ground_truth_canonical_to_source: np.ndarray,
    estimated_source_to_canonical: np.ndarray,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    columns: int = 9,
    rows: int = 13,
) -> AlignmentMetrics:
    """Measure recovery error at points that were not used to fit the markers."""
    grid = evaluation_grid(layout, columns, rows)
    observed = transform_points(grid, ground_truth_canonical_to_source)
    recovered = transform_points(observed, estimated_source_to_canonical)
    errors = np.linalg.norm(recovered - grid, axis=1)

    corners = np.asarray(layout.page_corners_px, dtype=np.float32)
    observed_corners = transform_points(corners, ground_truth_canonical_to_source)
    recovered_corners = transform_points(observed_corners, estimated_source_to_canonical)
    corner_errors = np.linalg.norm(recovered_corners - corners, axis=1)

    mean_px = float(np.mean(errors))
    rmse_px = float(np.sqrt(np.mean(errors**2)))
    p95_px = float(np.percentile(errors, 95))
    max_px = float(np.max(errors))
    corner_tuple = tuple(float(value) for value in corner_errors)
    return AlignmentMetrics(
        corner_errors_px=corner_tuple,
        corner_errors_mm=tuple(px_to_mm(value, layout.dpi) for value in corner_tuple),
        mean_error_px=mean_px,
        mean_error_mm=px_to_mm(mean_px, layout.dpi),
        rmse_px=rmse_px,
        rmse_mm=px_to_mm(rmse_px, layout.dpi),
        p95_error_px=p95_px,
        p95_error_mm=px_to_mm(p95_px, layout.dpi),
        max_error_px=max_px,
        max_error_mm=px_to_mm(max_px, layout.dpi),
        evaluation_point_count=int(errors.size),
    )


def draw_alignment_overlay(
    source_image: np.ndarray,
    ground_truth_canonical_to_source: np.ndarray,
    estimated_source_to_canonical: np.ndarray,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    columns: int = 9,
    rows: int = 13,
) -> np.ndarray:
    """Draw expected points in green and recovered model positions in magenta."""
    if source_image.ndim == 2:
        overlay = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = source_image.copy()
    grid = evaluation_grid(layout, columns, rows)
    expected = transform_points(grid, ground_truth_canonical_to_source)
    estimated_canonical_to_source = np.linalg.inv(estimated_source_to_canonical)
    recovered = transform_points(grid, estimated_canonical_to_source)
    radius = max(2, round(min(overlay.shape[:2]) / 500))
    for expected_point, recovered_point in zip(expected, recovered, strict=True):
        e = tuple(np.rint(expected_point).astype(int))
        r = tuple(np.rint(recovered_point).astype(int))
        cv2.line(overlay, e, r, (0, 190, 255), 1, cv2.LINE_AA)
        cv2.circle(overlay, e, radius, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, r, radius, (210, 0, 210), 1, cv2.LINE_AA)
    cv2.putText(overlay, "expected", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "recovered", (145, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 0, 210), 2, cv2.LINE_AA)
    return overlay
