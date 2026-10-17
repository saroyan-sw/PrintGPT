"""Old-to-old alignment before the strict registration check (deterministic, no network).

``fixtures/gemini_shift`` holds a real Gemini app output for the bottom strip of
a scanned test page (the source crop is stored at 75 DPI). Gemini returned the
strip with all original text moved about 1.1 mm right and 1.3 mm down.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from printgpt_alignment.geometry import mm_to_px
from printgpt_generation.extraction import ExtractionConfig, normalize_and_extract
from printgpt_generation.framing import PixelBox

from test_generation_fixtures import LAYOUT, make_masks, make_page


CONFIG = ExtractionConfig()
NO_ALIGNMENT = ExtractionConfig(auto_align=False)
FIXTURES = Path(__file__).parent / "fixtures" / "gemini_shift"


def _px(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, LAYOUT.dpi)))


def _similar(image: np.ndarray, *, shift_mm=(0.0, 0.0), scale=1.0, rotation_deg=0.0) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, scale)
    matrix[:, 2] += (mm_to_px(shift_mm[0], LAYOUT.dpi), mm_to_px(shift_mm[1], LAYOUT.dpi))
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def _drawing_window() -> tuple[np.ndarray, np.ndarray, np.ndarray, PixelBox]:
    """Synthetic page, its existing-ink mask, a window with one new stroke, and the window."""
    page, _ = make_page()
    masks = make_masks(page)
    box = PixelBox(_px(30), _px(80), _px(160), _px(120))
    window = box.crop(page).copy()
    cv2.line(window, (_px(70), _px(110)), (_px(120), _px(115)), (0, 0, 0), 3)
    return page, masks.existing_ink_mask, window, box


def _normalize(raw, existing, page, box, config=CONFIG):
    return normalize_and_extract(raw, existing, LAYOUT, config, source_page=page, canvas_box=box.as_tuple())


def test_unmoved_output_is_left_alone() -> None:
    page, existing, window, box = _drawing_window()
    outcome = _normalize(window, existing, page, box)
    assert outcome.accepted
    assert outcome.record.alignment is not None and not outcome.record.alignment.applied
    assert outcome.record.alignment.reason == "already aligned"


def test_shift_scale_and_rotation_are_corrected() -> None:
    page, existing, window, box = _drawing_window()
    moved = _similar(window, shift_mm=(1.5, -1.0), scale=1.03, rotation_deg=1.0)
    assert not _normalize(moved, existing, page, box, NO_ALIGNMENT).accepted
    outcome = _normalize(moved, existing, page, box)
    assert outcome.accepted, outcome.record.failure_reasons
    alignment = outcome.record.alignment
    assert alignment.applied
    assert abs(alignment.scale - 1.03) < 0.01
    assert abs(alignment.rotation_deg) == pytest.approx(1.0, abs=0.3)
    assert outcome.record.registration.estimated_shift_mm <= CONFIG.max_registration_shift_mm
    new_ink = outcome.candidate.candidate_new_ink_mask > 0
    # The new stroke lands where it was drawn relative to the original.
    expected = np.zeros(LAYOUT.image_shape, np.uint8)
    cv2.line(expected, (box.x + _px(70), box.y + _px(110)), (box.x + _px(120), box.y + _px(115)), 255, 3)
    near = cv2.dilate(expected, np.ones((5, 5), np.uint8)) > 0
    assert new_ink.any() and np.count_nonzero(new_ink & near) / np.count_nonzero(new_ink) > 0.9


@pytest.mark.parametrize(
    "change",
    [{"shift_mm": (8.0, 0.0)}, {"scale": 1.10}, {"rotation_deg": 5.0}],
    ids=["shift", "scale", "rotation"],
)
def test_movement_beyond_the_limits_is_not_corrected(change) -> None:
    page, existing, window, box = _drawing_window()
    outcome = _normalize(_similar(window, **change), existing, page, box)
    assert not outcome.accepted
    alignment = outcome.record.alignment
    assert alignment is not None and not alignment.applied


def test_uneven_distortion_is_still_rejected() -> None:
    page, existing, window, box = _drawing_window()
    height, width = window.shape[:2]
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    # Only the left part of the window moves (up to 2.5 mm); one similarity cannot fit both parts.
    bend = mm_to_px(2.5, LAYOUT.dpi) * np.clip(1.0 - cols / (0.5 * width), 0.0, 1.0)
    warped = cv2.remap(window, cols, rows - bend, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    outcome = _normalize(warped, existing, page, box)
    assert not outcome.accepted
    assert outcome.record.failure_reasons


def test_repeating_text_lines_are_not_aligned_a_whole_line_away() -> None:
    page = np.full((*LAYOUT.image_shape, 3), 255, np.uint8)
    for row in range(14):
        cv2.putText(page, "C:\\WINDOWS\\system32\\spool\\DRIVERS", (_px(20), _px(150 + 4.2 * row)),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    existing = np.where(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) < 200, 255, 0).astype(np.uint8)
    box = PixelBox(0, _px(140), LAYOUT.image_shape[1], _px(70))
    moved = _similar(box.crop(page).copy(), shift_mm=(0.0, 1.5))
    outcome = _normalize(moved, existing, page, box)
    assert outcome.accepted, outcome.record.failure_reasons
    shift_x, shift_y = outcome.record.alignment.centre_shift_mm
    assert abs(shift_y - 1.5) < 0.4 and abs(shift_x) < 0.4


def test_real_gemini_output_is_aligned_and_keeps_its_new_lines() -> None:
    source_crop = cv2.cvtColor(cv2.imread(str(FIXTURES / "source_crop_75dpi.png")), cv2.COLOR_BGR2RGB)
    raw = cv2.cvtColor(cv2.imread(str(FIXTURES / "gemini_output.jpg")), cv2.COLOR_BGR2RGB)
    height, width = source_crop.shape[:2]
    box = PixelBox(0, LAYOUT.image_shape[0] - height, width, height)
    page = np.full((*LAYOUT.image_shape, 3), 255, np.uint8)
    page[box.slices()] = source_crop
    existing = np.where(cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) < 200, 255, 0).astype(np.uint8)

    rejected = _normalize(raw, existing, page, box, NO_ALIGNMENT)
    assert not rejected.accepted
    assert rejected.record.registration.estimated_shift_mm > 1.0

    outcome = _normalize(raw, existing, page, box)
    assert outcome.accepted, outcome.record.failure_reasons
    alignment = outcome.record.alignment
    assert alignment.applied
    assert alignment.centre_shift_mm[0] == pytest.approx(1.1, abs=0.4)
    assert alignment.centre_shift_mm[1] == pytest.approx(1.3, abs=0.4)
    # At 75 DPI the small text is soft, so retention is lower than the 100% measured at 300 DPI.
    assert outcome.record.registration.source_retention >= CONFIG.min_source_retention
    # The five new lines Gemini drew sit in the lower part of the strip, below the original list.
    new_ink = outcome.candidate.candidate_new_ink_mask > 0
    rows = np.nonzero(new_ink.any(axis=1))[0]
    assert rows.size and rows.min() >= box.y + int(0.5 * height)
