"""Help the user place the allowed completion rectangle before any generation.

A rectangle typed blind in millimetres is easy to get wrong, and a wrong one
wastes paid model calls: the model continues the drawing where it belongs, and
Track A then rejects most of the new ink as outside the area. This module

- finds the existing drawing on the canonical page,
- suggests an area: the empty strip on the side where the drawing looks cut off
  (most strokes end at that edge), from the drawing's edge to the printable margin,
- checks a chosen rectangle and explains likely problems before generation, and
- draws the rectangle over the page preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from printgpt_alignment import DEFAULT_LAYOUT, A4CalibrationLayout


# Kept clear of the page edges: scanner edge lines and the printer's non-printable margin.
PAGE_MARGIN_MM: Final[float] = 6.0
MIN_STRIP_MM: Final[float] = 15.0
_EDGE_BAND_MM: Final[float] = 2.0
_INK_THRESHOLD: Final[int] = 128
_MAX_EXISTING_INK_SHARE: Final[float] = 0.25
# Marks smaller than this are dust or scanner noise, not part of the drawing.
_MIN_MARK_MM2: Final[float] = 1.5


@dataclass(frozen=True, slots=True)
class RectMM:
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def rounded(self) -> RectMM:
        return RectMM(*(float(round(value)) for value in (self.x, self.y, self.width, self.height)))

    def overlap(self, other: RectMM) -> float:
        width = max(0.0, min(self.right, other.right) - max(self.x, other.x))
        height = max(0.0, min(self.bottom, other.bottom) - max(self.y, other.y))
        return width * height


@dataclass(frozen=True, slots=True)
class RegionSuggestion:
    rect: RectMM
    side: str
    cut_strokes: int
    reason: str


def _px_per_mm(layout: A4CalibrationLayout) -> tuple[float, float]:
    return layout.raster_width_px / layout.page_width_mm, layout.raster_height_px / layout.page_height_mm


def ink_mask(page_rgb: np.ndarray, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """Dark pixels of the drawing, ignoring scanner edge lines and dust specks."""
    ink = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY) < _INK_THRESHOLD
    sx, sy = _px_per_mm(layout)
    border_x, border_y = int(round(3 * sx)), int(round(3 * sy))
    ink[:border_y] = ink[-border_y:] = False
    ink[:, :border_x] = ink[:, -border_x:] = False
    # Specks far from the drawing would otherwise stretch its bounds across the page.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= _MIN_MARK_MM2 * sx * sy
    return keep[labels]


def drawing_bounds(page_rgb: np.ndarray, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> RectMM | None:
    ink = ink_mask(page_rgb, layout)
    points = cv2.findNonZero(ink.astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    sx, sy = _px_per_mm(layout)
    return RectMM(x / sx, y / sy, width / sx, height / sy)


def _strokes_crossing(ink: np.ndarray, band: tuple[slice, slice]) -> int:
    count, _ = cv2.connectedComponents(ink[band].astype(np.uint8), connectivity=8)
    return count - 1


def suggest_region(page_rgb: np.ndarray, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> RegionSuggestion | None:
    """Empty strip beside the drawing on the side where it looks cut off; ``None`` if no room."""
    bounds = drawing_bounds(page_rgb, layout)
    page_w, page_h = layout.page_width_mm, layout.page_height_mm
    lo_x, lo_y, hi_x, hi_y = PAGE_MARGIN_MM, PAGE_MARGIN_MM, page_w - PAGE_MARGIN_MM, page_h - PAGE_MARGIN_MM
    if bounds is None:
        return RegionSuggestion(RectMM(lo_x, lo_y, hi_x - lo_x, hi_y - lo_y), "page", 0, "the page is empty")
    ink = ink_mask(page_rgb, layout)
    sx, sy = _px_per_mm(layout)
    band_x, band_y = max(1, int(round(_EDGE_BAND_MM * sx))), max(1, int(round(_EDGE_BAND_MM * sy)))
    x0, y0 = int(round(bounds.x * sx)), int(round(bounds.y * sy))
    x1, y1 = int(round(bounds.right * sx)), int(round(bounds.bottom * sy))
    # Along the drawing's other axis, the strip keeps a little room beyond the drawing.
    span_y = (max(lo_y, bounds.y - 10), min(hi_y, bounds.bottom + 10))
    span_x = (max(lo_x, bounds.x - 10), min(hi_x, bounds.right + 10))
    sides = {
        "right": (RectMM(bounds.right, span_y[0], hi_x - bounds.right, span_y[1] - span_y[0]),
                  (slice(y0, y1), slice(max(x0, x1 - band_x), x1))),
        "left": (RectMM(lo_x, span_y[0], bounds.x - lo_x, span_y[1] - span_y[0]),
                 (slice(y0, y1), slice(x0, min(x1, x0 + band_x)))),
        "below": (RectMM(span_x[0], bounds.bottom, span_x[1] - span_x[0], hi_y - bounds.bottom),
                  (slice(max(y0, y1 - band_y), y1), slice(x0, x1))),
        "above": (RectMM(span_x[0], lo_y, span_x[1] - span_x[0], bounds.y - lo_y),
                  (slice(y0, min(y1, y0 + band_y)), slice(x0, x1))),
    }
    options = []
    for side, (rect, band) in sides.items():
        if rect.width >= MIN_STRIP_MM and rect.height >= MIN_STRIP_MM:
            options.append((_strokes_crossing(ink, band), rect.width * rect.height, side, rect))
    if not options:
        return None
    cut, _, side, rect = max(options)
    reason = (f"{cut} strokes end at the drawing's {side} edge, so it looks cut off there" if cut > 2
              else f"largest empty space next to the drawing is {side} of it")
    return RegionSuggestion(rect.rounded(), side, cut, reason)


def check_region(page_rgb: np.ndarray, rect: RectMM, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> list[str]:
    """Plain-language problems with ``rect``; an empty list means nothing obvious is wrong."""
    problems = []
    page = RectMM(0, 0, layout.page_width_mm, layout.page_height_mm)
    if rect.width < MIN_STRIP_MM or rect.height < MIN_STRIP_MM:
        problems.append(f"The area is very small ({rect.width:.0f} x {rect.height:.0f} mm); there is little room to draw.")
    if rect.x < PAGE_MARGIN_MM or rect.y < PAGE_MARGIN_MM or rect.right > page.width - PAGE_MARGIN_MM \
            or rect.bottom > page.height - PAGE_MARGIN_MM:
        problems.append(f"The area reaches within {PAGE_MARGIN_MM:.0f} mm of the paper edge, where the printer "
                        "cannot print; ink there will be cut.")
    ink = ink_mask(page_rgb, layout)
    total = int(np.count_nonzero(ink))
    if total:
        sx, sy = _px_per_mm(layout)
        window = ink[int(rect.y * sy):int(np.ceil(rect.bottom * sy)), int(rect.x * sx):int(np.ceil(rect.right * sx))]
        share = np.count_nonzero(window) / total
        if share > _MAX_EXISTING_INK_SHARE:
            problems.append(f"The area covers {share:.0%} of the existing drawing. New ink cannot go on or next to "
                            "existing lines, so most of the area is unusable.")
    suggestion = suggest_region(page_rgb, layout)
    if suggestion is not None and suggestion.cut_strokes > 2:
        covered = rect.overlap(suggestion.rect) / max(suggestion.rect.width * suggestion.rect.height, 1e-9)
        if covered < 0.5:
            problems.append(
                f"The drawing looks cut off on the {suggestion.side} ({suggestion.cut_strokes} strokes end there), "
                f"but the area covers only {covered:.0%} of the space on that side. The continuation will "
                "probably be drawn outside the area and rejected."
            )
    return problems


def draw_region(page_rgb: np.ndarray, rect: RectMM, layout: A4CalibrationLayout = DEFAULT_LAYOUT,
                suggestion: RectMM | None = None) -> np.ndarray:
    """Page preview with the chosen area tinted green and outlined; a suggestion is outlined in blue."""
    sx, sy = _px_per_mm(layout)
    preview = page_rgb.copy()
    thickness = max(2, int(round(0.6 * sx)))

    def corners(area: RectMM) -> tuple[tuple[int, int], tuple[int, int]]:
        return (int(area.x * sx), int(area.y * sy)), (int(area.right * sx), int(area.bottom * sy))

    top_left, bottom_right = corners(rect)
    tinted = preview.copy()
    cv2.rectangle(tinted, top_left, bottom_right, (120, 220, 120), -1)
    preview = cv2.addWeighted(tinted, 0.3, preview, 0.7, 0)
    if suggestion is not None and suggestion != rect:
        cv2.rectangle(preview, *corners(suggestion), (40, 90, 255), thickness)
    cv2.rectangle(preview, top_left, bottom_right, (230, 0, 230), thickness)
    return preview
