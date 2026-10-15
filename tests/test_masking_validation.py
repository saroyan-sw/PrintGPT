import numpy as np
import pytest

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_contracts import MaskingConfig, ValidationConfig
from printgpt_masking import build_masks, validate_candidate


LAYOUT = A4CalibrationLayout(dpi=25.4)
LENIENT = ValidationConfig(
    min_final_ink_pixels=1,
    max_final_coverage_fraction=1.0,
    max_removed_fraction=1.0,
    max_small_component_fraction=1.0,
)


def _page() -> np.ndarray:
    return np.full((*LAYOUT.image_shape, 3), 255, dtype=np.uint8)


def _mask(fill: int = 0) -> np.ndarray:
    return np.full(LAYOUT.image_shape, fill, dtype=np.uint8)


def _config(**overrides: object) -> MaskingConfig:
    values: dict[str, object] = {
        "clearance_mm": 0.0,
        "zero_clearance_test_mode": True,
        "printer_margin_mm": 0.0,
        "layout_guide_clearance_mm": 0.0,
        "exclude_calibration_labels": False,
    }
    values.update(overrides)
    return MaskingConfig(**values)  # type: ignore[arg-type]


def _bundle(allowed: np.ndarray | None = None, **config: object):
    return build_masks(_page(), _mask(255) if allowed is None else allowed, layout=LAYOUT, config=_config(**config))


@pytest.mark.parametrize(
    "bad_page",
    [
        np.zeros(LAYOUT.image_shape, dtype=np.uint8),
        np.zeros((*LAYOUT.image_shape, 4), dtype=np.uint8),
        np.zeros((*LAYOUT.image_shape, 3), dtype=np.float32),
    ],
)
def test_malformed_page_is_rejected(bad_page: np.ndarray) -> None:
    with pytest.raises(ValueError):
        build_masks(bad_page, _mask(255), layout=LAYOUT, config=_config())


@pytest.mark.parametrize(
    "bad_mask",
    [
        None,
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros(LAYOUT.image_shape, dtype=bool),
        np.full(LAYOUT.image_shape, 1, dtype=np.uint8),
    ],
)
def test_missing_or_malformed_allowed_region_is_rejected(bad_mask: np.ndarray | None) -> None:
    with pytest.raises(ValueError):
        build_masks(_page(), bad_mask, layout=LAYOUT, config=_config())  # type: ignore[arg-type]


def test_malformed_candidate_is_rejected_without_resizing() -> None:
    masks = _bundle()
    with pytest.raises(ValueError, match="shape"):
        validate_candidate(np.zeros((10, 10), dtype=np.uint8), masks, config=LENIENT)
    invalid = _mask()
    invalid[10, 10] = 1
    with pytest.raises(ValueError, match="exactly 0 or 255"):
        validate_candidate(invalid, masks, config=LENIENT)


def test_protected_area_is_clipped_and_reported() -> None:
    page = _page()
    page[100, 100] = 0
    masks = build_masks(page, _mask(255), layout=LAYOUT, config=_config())
    candidate = _mask()
    candidate[100, 100:110] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert result.metrics["protected_content_overlap_pixels"] == 1
    assert result.sanitized_new_ink_mask[100, 100] == 0
    assert result.sanitized_new_ink_mask[100, 101] == 255


def test_allowed_region_clipping_is_measured() -> None:
    allowed = _mask()
    allowed[50:100, 50:100] = 255
    masks = _bundle(allowed)
    candidate = _mask()
    candidate[60, 60] = 255
    candidate[120, 120] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert result.metrics["outside_allowed_region_pixels"] == 1
    assert result.metrics["removed_pixel_count"] == 1


def test_page_margin_clipping_is_measured_in_millimetres() -> None:
    masks = _bundle(printer_margin_mm=10.0)
    candidate = _mask()
    candidate[50, 5] = 255
    candidate[50, 20] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert result.metrics["margin_violation_pixels"] == 1
    assert result.sanitized_new_ink_mask[50, 5] == 0
    assert result.sanitized_new_ink_mask[50, 20] == 255


def test_marker_and_quiet_zone_are_excluded() -> None:
    masks = _bundle()
    candidate = _mask()
    candidate[19, 19] = 255  # marker interior
    candidate[10, 10] = 255  # 3 mm quiet zone around the 12 mm inset
    candidate[50, 50] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert result.metrics["layout_violation_pixels"] == 2
    assert result.sanitized_new_ink_mask[19, 19] == 0
    assert result.sanitized_new_ink_mask[10, 10] == 0
    assert result.sanitized_new_ink_mask[50, 50] == 255


def test_calibration_labels_and_layout_guides_are_excluded() -> None:
    masks = _bundle(layout_guide_clearance_mm=1.0, exclude_calibration_labels=True)
    candidate = _mask()
    candidate[16, 100] = 255  # central top calibration label
    candidate[100, 10] = 255  # protected-content guide
    candidate[100, 100] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert result.metrics["layout_violation_pixels"] == 2
    assert result.sanitized_new_ink_mask[16, 100] == 0
    assert result.sanitized_new_ink_mask[100, 10] == 0
    assert result.sanitized_new_ink_mask[100, 100] == 255


def test_zero_allowed_area_is_rejected_with_explicit_zero_denominators() -> None:
    masks = _bundle(_mask())
    result = validate_candidate(_mask(), masks)
    assert not result.accepted
    assert "zero area" in result.reasons[0]
    assert result.metrics["final_coverage"] == 0.0
    assert result.metrics["removed_fraction"] == 0.0


def test_nearly_empty_addition_is_rejected() -> None:
    candidate = _mask()
    candidate[100, 100] = 255
    result = validate_candidate(candidate, _bundle(), config=ValidationConfig(min_final_ink_pixels=2))
    assert not result.accepted
    assert any("below minimum" in reason for reason in result.reasons)


def test_excessive_coverage_is_rejected() -> None:
    masks = _bundle()
    candidate = masks.printable_region_mask.copy()
    result = validate_candidate(
        candidate,
        masks,
        config=ValidationConfig(min_final_ink_pixels=1, max_final_coverage_fraction=0.10),
    )
    assert not result.accepted
    assert result.metrics["final_coverage"] == 1.0


def test_excessive_clipping_is_rejected_even_when_remaining_pixels_are_safe() -> None:
    allowed = _mask()
    allowed[:, :105] = 255
    candidate = _mask()
    candidate[100, 90:120] = 255
    result = validate_candidate(
        candidate,
        _bundle(allowed),
        config=ValidationConfig(
            min_final_ink_pixels=1,
            max_final_coverage_fraction=1.0,
            max_removed_fraction=0.25,
            max_small_component_fraction=1.0,
        ),
    )
    assert not result.accepted
    assert result.metrics["removed_fraction"] == 0.5


def test_scattered_noise_is_rejected() -> None:
    candidate = _mask()
    for y, x in ((50, 50), (80, 80), (110, 110), (140, 140)):
        candidate[y, x] = 255
    result = validate_candidate(
        candidate,
        _bundle(),
        config=ValidationConfig(
            min_final_ink_pixels=1,
            max_final_coverage_fraction=1.0,
            max_removed_fraction=1.0,
            small_component_max_pixels=1,
            max_small_component_fraction=0.5,
        ),
    )
    assert not result.accepted
    assert result.metrics["connected_component_count"] == 4
    assert result.metrics["small_component_fraction"] == 1.0


def test_positive_clearance_exposes_endpoint_gap() -> None:
    page = _page()
    page[100, 80:101] = 0
    masks = build_masks(
        page,
        _mask(255),
        layout=LAYOUT,
        config=MaskingConfig(
            clearance_mm=2.0,
            printer_margin_mm=0.0,
            layout_guide_clearance_mm=0.0,
            exclude_calibration_labels=False,
        ),
    )
    candidate = _mask()
    candidate[100, 101:111] = 255
    result = validate_candidate(candidate, masks, config=LENIENT)
    assert np.all(result.sanitized_new_ink_mask[100, 101:103] == 0)
    assert result.sanitized_new_ink_mask[100, 103] == 255
    assert result.metrics["removed_pixel_count"] == 2


def test_sanitized_mask_has_zero_intersection_with_all_forbidden_regions() -> None:
    masks = _bundle(printer_margin_mm=10.0, layout_guide_clearance_mm=1.0)
    candidate = _mask(255)
    result = validate_candidate(candidate, masks, config=LENIENT)
    forbidden = (
        (masks.allowed_region_mask == 0)
        | (masks.safe_page_interior_mask == 0)
        | (masks.protected_mask > 0)
        | (masks.layout_exclusion_mask > 0)
    )
    assert not np.any((result.sanitized_new_ink_mask > 0) & forbidden)
    assert result.metrics["final_forbidden_overlap_pixels"] == 0


def test_metrics_include_mm2_areas_thresholds_and_denominators() -> None:
    candidate = _mask()
    candidate[100, 100:105] = 255
    result = validate_candidate(candidate, _bundle(), config=LENIENT)
    assert result.metrics["areas_mm2"]["proposed_ink"] == 5.0  # type: ignore[index]
    assert result.metrics["validation_thresholds"] == LENIENT.to_dict()
    assert "removed_fraction" in result.metrics["fraction_denominators"]  # type: ignore[operator]


def test_repeated_validation_is_deterministic() -> None:
    candidate = _mask()
    candidate[120, 80:120] = 255
    original = candidate.copy()
    masks = _bundle()
    first = validate_candidate(candidate, masks, config=LENIENT)
    second = validate_candidate(candidate, masks, config=LENIENT)
    assert first.accepted == second.accepted
    assert first.metrics == second.metrics
    assert np.array_equal(first.sanitized_new_ink_mask, second.sanitized_new_ink_mask)
    assert np.array_equal(candidate, original)
