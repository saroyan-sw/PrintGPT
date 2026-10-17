"""Allowed-area suggestion and pre-flight checks (deterministic pages, no generation)."""

import cv2
import numpy as np
import pytest

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_mvp.region_helper import PAGE_MARGIN_MM, RectMM, check_region, draw_region, drawing_bounds, suggest_region

LAYOUT = A4CalibrationLayout(dpi=75.0)
PX = LAYOUT.dpi / 25.4


def _page() -> np.ndarray:
    return np.full((*LAYOUT.image_shape, 3), 255, np.uint8)


def _p(x_mm: float, y_mm: float) -> tuple[int, int]:
    return int(round(x_mm * PX)), int(round(y_mm * PX))


def _cut_on_right(cut_x_mm: float = 120) -> np.ndarray:
    """Like the half cat: many strokes end abruptly on one vertical line."""
    page = _page()
    for y in range(40, 200, 15):
        cv2.line(page, _p(30, y), _p(cut_x_mm, y + 3), (0, 0, 0), 2)
    cv2.line(page, _p(30, 40), _p(30, 195), (0, 0, 0), 2)
    return page


def test_cut_side_is_suggested_from_the_cut_to_the_margin() -> None:
    suggestion = suggest_region(_cut_on_right(), LAYOUT)
    assert suggestion is not None and suggestion.side == "right" and suggestion.cut_strokes > 5
    rect = suggestion.rect
    assert rect.x == pytest.approx(120, abs=1.5)
    assert rect.right == pytest.approx(210 - PAGE_MARGIN_MM, abs=1)
    assert check_region(_cut_on_right(), rect, LAYOUT) == []


def test_cut_at_the_bottom_suggests_below() -> None:
    page = _page()
    for x in range(40, 170, 12):
        cv2.line(page, _p(x, 30), _p(x + 4, 140), (0, 0, 0), 2)
    suggestion = suggest_region(page, LAYOUT)
    assert suggestion.side == "below"
    assert suggestion.rect.y == pytest.approx(140, abs=1.5)


def test_complete_drawing_gets_the_largest_empty_space() -> None:
    page = _page()
    cv2.circle(page, _p(60, 60), int(25 * PX), (0, 0, 0), 2)
    suggestion = suggest_region(page, LAYOUT)
    assert suggestion.cut_strokes <= 2 and suggestion.side in ("right", "below")
    assert "largest empty space" in suggestion.reason


def test_empty_page_suggests_the_printable_page() -> None:
    suggestion = suggest_region(_page(), LAYOUT)
    assert suggestion.side == "page" and drawing_bounds(_page(), LAYOUT) is None


def test_scanner_edge_lines_are_ignored() -> None:
    page = _cut_on_right()
    page[:2] = 0  # dark scanner edge line at the very top
    assert drawing_bounds(page, LAYOUT).y == pytest.approx(40, abs=1.5)


def test_wrong_area_gets_specific_warnings() -> None:
    page = _cut_on_right()
    problems = check_region(page, RectMM(20, 30, 100, 170), LAYOUT)
    assert any("of the existing drawing" in problem for problem in problems)
    assert any("cut off on the right" in problem for problem in problems)


def test_tiny_or_edge_areas_are_flagged() -> None:
    page = _page()
    assert any("very small" in p for p in check_region(page, RectMM(100, 100, 8, 8), LAYOUT))
    assert any("paper edge" in p for p in check_region(page, RectMM(2, 100, 50, 50), LAYOUT))


def test_preview_marks_area_and_suggestion() -> None:
    page = _cut_on_right()
    preview = draw_region(page, RectMM(20, 30, 100, 170), LAYOUT, suggestion=RectMM(120, 30, 84, 180))
    assert preview.shape == page.shape and not np.array_equal(preview, page)
    x, y = _p(150, 100)
    assert tuple(preview[y, x]) == (255, 255, 255)  # outside the area stays untinted
    x, y = _p(60, 35)
    assert preview[y, x, 1] > preview[y, x, 0]  # inside the area is tinted green
