"""Bounded old-to-old alignment of provider output before the registration check.

Image models that redraw the page often move the whole of it slightly. Their
new strokes are drawn relative to their own copy of the original, so mapping
that copy back onto the real original also puts the new strokes where the
model meant them relative to the page.

The correction is a similarity transform (shift, uniform scale, small
rotation) estimated from the original ink in both images. It is only a
correction step: the strict registration check in ``extraction`` still runs
afterwards, unchanged. Uneven distortion cannot be fitted by one similarity
transform and is therefore still rejected there.

Estimation works on downsampled, blurred ink masks so line art has a smooth
basin to converge in, and only near the original ink, so new content (which has
no counterpart in the source) cannot pull the fit. It starts from no movement
and from the phase-correlation shift, and prefers the smaller movement when
both fit about equally well, so a page of evenly spaced lines cannot be
"aligned" a whole line away. A pure shift is preferred unless adding scale and
rotation fits clearly better.
"""

from __future__ import annotations

from dataclasses import replace
from math import atan2, degrees, hypot

import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout, px_to_mm

from .models import AlignmentReport


# Corrections below this (or below one pixel) are resampling noise, not movement.
_NEGLIGIBLE_MM = 0.1
# Correlations closer than this count as equally good; the smaller movement wins.
_CORRELATION_TIE = 0.01
# Scale and rotation are used only when they fit at least this much better than a pure shift.
_MODEL_GAIN = 0.02
_ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)


def downsample_mask(mask: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Float copy of ``mask`` whose longer side is at most ``max_side``, and the factor used."""
    factor = max(mask.shape) / float(max_side)
    if factor <= 1.0:
        return mask.astype(np.float32), 1.0
    size = (int(round(mask.shape[1] / factor)), int(round(mask.shape[0] / factor)))
    return cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA), factor


def _not_applied(reason: str, correlation: float = 0.0) -> AlignmentReport:
    return AlignmentReport(False, (0.0, 0.0), 0.0, 1.0, 0.0, 0.0, correlation, reason)


def _refine(
    template: np.ndarray, image: np.ndarray, initial: np.ndarray, motion: int, mask: np.ndarray
) -> tuple[float, np.ndarray] | None:
    warp = initial.astype(np.float32).copy()
    try:
        correlation, warp = cv2.findTransformECC(template, image, warp, motion, _ECC_CRITERIA, mask, 1)
    except cv2.error:
        return None
    return float(correlation), warp


def _similarity(warp: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Nearest similarity to an affine warp: (2x3 matrix, scale, rotation_deg, anisotropy)."""
    a, b, c, d = float(warp[0, 0]), float(warp[0, 1]), float(warp[1, 0]), float(warp[1, 1])
    p, q = (a + d) / 2.0, (c - b) / 2.0
    scale = hypot(p, q)
    linear = np.array([[p, -q], [q, p]], dtype=np.float64)
    anisotropy = float(np.linalg.norm(np.array([[a, b], [c, d]]) - linear) / max(scale, 1e-9))
    matrix = np.hstack([linear, warp[:, 2:3].astype(np.float64)])
    return matrix, scale, degrees(atan2(q, p)), anisotropy


def _max_corner_displacement_px(matrix: np.ndarray, shape: tuple[int, ...]) -> float:
    height, width = shape[:2]
    corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float64)
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    return float(np.max(np.hypot(*(moved - corners).T)))


def estimate_alignment(
    source_ink: np.ndarray,
    candidate_ink: np.ndarray,
    layout: A4CalibrationLayout,
    *,
    max_side_px: int,
    blur_mm: float,
    max_shift_mm: float,
    max_scale_change: float,
    max_rotation_deg: float,
    max_anisotropy: float,
    min_source_pixels: int,
) -> tuple[np.ndarray | None, AlignmentReport]:
    """Estimate the correction that maps original ink in ``candidate_ink`` onto ``source_ink``.

    Returns a full-resolution 2x3 matrix for ``cv2.warpAffine`` with
    ``WARP_INVERSE_MAP`` (corrected pixel -> candidate pixel), or ``None`` when no
    correction should be applied; the report always says why.
    """
    if int(np.count_nonzero(source_ink)) < min_source_pixels:
        return None, _not_applied("too little original ink to align on")
    if not candidate_ink.any():
        return None, _not_applied("candidate has no dark ink")

    template, factor = downsample_mask(source_ink, max_side_px)
    image, _ = downsample_mask(candidate_ink, max_side_px)
    sigma = max(1.0, blur_mm * layout.dpi / 25.4 / factor)
    template = cv2.GaussianBlur(template, (0, 0), sigma)
    image = cv2.GaussianBlur(image, (0, 0), sigma)

    # Compare only near the original ink; new strokes elsewhere have no counterpart.
    support = (template > 0.02).astype(np.uint8)
    radius = int(np.ceil(3 * sigma))
    mask = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))

    (shift_x, shift_y), _ = cv2.phaseCorrelate(template, image)
    starts = [np.eye(2, 3), np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])]

    def movement(fit: tuple[float, np.ndarray]) -> float:
        return _max_corner_displacement_px(_similarity(fit[1])[0], template.shape)

    def best(motion: int) -> tuple[float, np.ndarray] | None:
        fits = [fit for fit in (_refine(template, image, start, motion, mask) for start in starts) if fit]
        if not fits:
            return None
        top = max(correlation for correlation, _ in fits)
        return min((fit for fit in fits if fit[0] >= top - _CORRELATION_TIE), key=movement)

    shift_fit, affine_fit = best(cv2.MOTION_TRANSLATION), best(cv2.MOTION_AFFINE)
    if shift_fit is None and affine_fit is None:
        return None, _not_applied("alignment did not converge")
    if affine_fit is not None and (shift_fit is None or affine_fit[0] > shift_fit[0] + _MODEL_GAIN):
        correlation, warp = affine_fit
    else:
        assert shift_fit is not None
        correlation, warp = shift_fit
    small, scale, rotation, anisotropy = _similarity(warp)
    matrix = small.copy()
    matrix[:, 2] *= factor
    height, width = source_ink.shape[:2]
    centre = np.array([width / 2.0, height / 2.0])
    centre_shift = centre @ matrix[:, :2].T + matrix[:, 2] - centre
    displacement_mm = px_to_mm(_max_corner_displacement_px(matrix, source_ink.shape), layout.dpi)
    shift_mm = px_to_mm(float(np.hypot(*centre_shift)), layout.dpi)
    report = AlignmentReport(
        applied=False,
        centre_shift_mm=(px_to_mm(float(centre_shift[0]), layout.dpi), px_to_mm(float(centre_shift[1]), layout.dpi)),
        max_displacement_mm=displacement_mm,
        scale=scale,
        rotation_deg=rotation,
        anisotropy=anisotropy,
        correlation=correlation,
        reason="",
    )

    limits = (
        # Scale and rotation have their own limits, so the shift limit applies at the window centre.
        (shift_mm > max_shift_mm, f"shift {shift_mm:.2f} mm exceeds {max_shift_mm:.2f} mm"),
        (abs(scale - 1.0) > max_scale_change, f"scale {scale:.3f} exceeds +/-{max_scale_change:.0%}"),
        (abs(rotation) > max_rotation_deg, f"rotation {rotation:.2f} deg exceeds {max_rotation_deg:.1f} deg"),
        (anisotropy > max_anisotropy, f"distortion {anisotropy:.3f} is not a shift, scale or rotation"),
    )
    for exceeded, reason in limits:
        if exceeded:
            return None, replace(report, reason=f"not applied: {reason}")
    if displacement_mm < max(_NEGLIGIBLE_MM, px_to_mm(1.0, layout.dpi)):
        return None, replace(report, reason="already aligned")
    return matrix, replace(report, applied=True, reason="applied")


def apply_alignment(image: np.ndarray, matrix: np.ndarray, *, nearest: bool = False) -> np.ndarray:
    """Resample ``image`` so its original content lands on the source; uncovered border is paper."""
    height, width = image.shape[:2]
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    fill = (255, 255, 255) if image.ndim == 3 else 0
    return cv2.warpAffine(
        image, matrix.astype(np.float64), (width, height),
        flags=interpolation | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=fill,
    )


def as_homogeneous(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([matrix.astype(np.float64), [0.0, 0.0, 1.0]])
