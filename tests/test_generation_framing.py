"""Crop windows, aspect fitting, and crop-framed normalization (deterministic, no network)."""

import cv2
import numpy as np
import pytest

from printgpt_alignment.geometry import mm_to_px
from printgpt_generation.extraction import ExtractionConfig, normalize_and_extract
from printgpt_generation.framing import (
    PixelBox,
    best_aspect_frame,
    expand,
    fit_aspect,
    full_box,
    mask_bounds,
    parse_aspect,
)

from test_generation_fixtures import LAYOUT, make_masks, make_page


SHAPE = LAYOUT.image_shape
CONFIG = ExtractionConfig()


def _px(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, LAYOUT.dpi)))


def _contains(outer: PixelBox, inner: PixelBox) -> bool:
    return outer.x <= inner.x and outer.y <= inner.y and outer.right >= inner.right and outer.bottom >= inner.bottom


@pytest.mark.parametrize("aspect", ["1:1", "3:2", "16:9", "21:9", "9:16", "3:4"])
@pytest.mark.parametrize("box", [PixelBox(50, 700, 420, 90), PixelBox(0, 0, 30, 30), PixelBox(500, 820, 100, 50)])
def test_fit_aspect_contains_the_box_and_matches_the_ratio(box: PixelBox, aspect: str) -> None:
    frame = fit_aspect(box, aspect, SHAPE)
    assert _contains(frame.box, box)
    assert frame.box.fits(SHAPE)
    assert abs(frame.canvas_width / frame.canvas_height / parse_aspect(aspect) - 1) < 0.02
    pad_right, pad_bottom = frame.padding
    assert pad_right >= 0 and pad_bottom >= 0
    # Padding only appears when the page itself is too small in that direction.
    assert pad_right == 0 or frame.box.width == SHAPE[1]
    assert pad_bottom == 0 or frame.box.height == SHAPE[0]


def test_full_page_at_three_by_four_pads_only_on_the_right() -> None:
    frame = fit_aspect(full_box(SHAPE), "3:4", SHAPE)
    assert frame.box == full_box(SHAPE)
    assert frame.padding[1] == 0 and frame.padding[0] > 0


def test_best_aspect_prefers_the_smallest_canvas() -> None:
    wide = PixelBox(40, 600, 500, 100)
    assert best_aspect_frame(wide, ("1:1", "3:4", "21:9"), SHAPE).aspect == "21:9"
    with pytest.raises(ValueError):
        best_aspect_frame(wide, (), SHAPE)


def test_mask_bounds_and_expand_clip_to_the_page() -> None:
    mask = np.zeros(SHAPE, np.uint8)
    assert mask_bounds(mask) is None
    mask[10:20, 30:50] = 255
    bounds = mask_bounds(mask)
    assert bounds == PixelBox(30, 10, 20, 10)
    grown = expand(bounds, 100, SHAPE)
    assert grown.x == 0 and grown.y == 0 and grown.right == 150 and grown.bottom == 120
    with pytest.raises(ValueError):
        PixelBox(0, 0, 0, 5)


def _crop_output(shift_mm: float = 0.0) -> tuple[np.ndarray, PixelBox, np.ndarray, np.ndarray]:
    """A crop of the page with one new stroke, upscaled as a provider would return it."""
    page, _ = make_page()
    masks = make_masks(page)
    box = PixelBox(_px(30), _px(80), _px(160), _px(120))
    window = box.crop(page).copy()
    start, end = (_px(70), _px(110)), (_px(120), _px(115))
    cv2.line(window, start, end, (0, 0, 0), 3)
    if shift_mm:
        shift = _px(shift_mm)
        window = cv2.warpAffine(window, np.float32([[1, 0, shift], [0, 1, 0]]), window.shape[1::-1],
                                borderValue=(255, 255, 255))
    raw = cv2.resize(window, (box.width * 2, box.height * 2), interpolation=cv2.INTER_LINEAR)
    return raw, box, page, masks.existing_ink_mask


def test_crop_output_is_pasted_into_its_window() -> None:
    raw, box, page, existing = _crop_output()
    outcome = normalize_and_extract(raw, existing, LAYOUT, CONFIG, source_page=page, canvas_box=box.as_tuple())
    assert outcome.accepted, outcome.record.failure_reasons
    assert outcome.record.canvas_box == box.as_tuple()
    assert outcome.record.raw_to_canonical[0] == (0.5, 0.0, float(box.x))
    assert outcome.record.registration is not None and outcome.record.registration.verified
    candidate = outcome.candidate
    assert candidate is not None and candidate.candidate_rgb.shape == (*SHAPE, 3)
    outside = np.ones(SHAPE, bool)
    outside[box.slices()] = False
    assert np.array_equal(candidate.candidate_rgb[outside], page[outside])
    new_ink = candidate.candidate_new_ink_mask > 0
    assert new_ink.any() and not new_ink[outside].any()
    assert not np.any(new_ink & (existing > 0))


def test_shift_inside_the_window_is_rejected_although_the_page_is_unchanged() -> None:
    raw, box, page, existing = _crop_output(shift_mm=3.0)
    no_alignment = ExtractionConfig(auto_align=False)
    outcome = normalize_and_extract(raw, existing, LAYOUT, no_alignment, source_page=page, canvas_box=box.as_tuple())
    assert not outcome.accepted
    assert outcome.record.canvas_box == box.as_tuple()
    assert outcome.record.registration is not None and outcome.record.registration.estimated_shift_mm > 2.0


def test_shift_inside_the_window_is_corrected_before_the_strict_check() -> None:
    raw, box, page, existing = _crop_output(shift_mm=3.0)
    outcome = normalize_and_extract(raw, existing, LAYOUT, CONFIG, source_page=page, canvas_box=box.as_tuple())
    assert outcome.accepted, outcome.record.failure_reasons
    alignment = outcome.record.alignment
    assert alignment is not None and alignment.applied
    assert abs(alignment.centre_shift_mm[0] - 3.0) < 0.4 and abs(alignment.centre_shift_mm[1]) < 0.4
    assert outcome.record.registration.estimated_shift_mm <= CONFIG.max_registration_shift_mm
    # The recorded transform includes the correction: raw x maps 3 mm further left.
    assert abs(outcome.record.raw_to_canonical[0][2] - (box.x - _px(3.0))) < 2


def test_crop_output_must_match_the_window_aspect() -> None:
    raw, box, page, existing = _crop_output()
    squashed = cv2.resize(raw, (raw.shape[1], raw.shape[0] // 2))
    outcome = normalize_and_extract(squashed, existing, LAYOUT, CONFIG, source_page=page, canvas_box=box.as_tuple())
    assert not outcome.accepted
    assert "declared" in outcome.record.failure_reasons[0]


def test_crop_outside_the_page_is_rejected_and_source_is_required() -> None:
    raw, box, page, existing = _crop_output()
    outside = (SHAPE[1] - 10, 0, box.width, box.height)
    outcome = normalize_and_extract(raw, existing, LAYOUT, CONFIG, source_page=page, canvas_box=outside)
    assert not outcome.accepted and "outside" in outcome.record.failure_reasons[0]
    with pytest.raises(ValueError):
        normalize_and_extract(raw, existing, LAYOUT, CONFIG, canvas_box=box.as_tuple())
