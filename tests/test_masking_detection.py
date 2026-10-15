import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_contracts import MaskingConfig
from printgpt_masking import build_masks, detect_existing_ink


LAYOUT = A4CalibrationLayout(dpi=25.4)


def _white_page() -> np.ndarray:
    return np.full((*LAYOUT.image_shape, 3), 255, dtype=np.uint8)


def _all_allowed() -> np.ndarray:
    return np.full(LAYOUT.image_shape, 255, dtype=np.uint8)


def _zero_clearance_config(**overrides: object) -> MaskingConfig:
    values: dict[str, object] = {
        "absolute_ink_threshold": 220,
        "local_contrast_threshold": 10,
        "background_blur_mm": 2.0,
        "clearance_mm": 0.0,
        "zero_clearance_test_mode": True,
        "printer_margin_mm": 0.0,
        "layout_guide_clearance_mm": 0.0,
        "exclude_calibration_labels": False,
    }
    values.update(overrides)
    return MaskingConfig(**values)  # type: ignore[arg-type]


def test_detects_known_dark_faint_and_thin_strokes() -> None:
    page = _white_page()
    cv2.line(page, (50, 80), (140, 80), (0, 0, 0), 2)
    cv2.line(page, (50, 100), (140, 100), (215, 215, 215), 1)
    mask = detect_existing_ink(page, layout=LAYOUT, config=_zero_clearance_config())
    assert mask[80, 75] == 255
    assert mask[100, 75] == 255
    assert mask[120, 75] == 0


def test_local_contrast_handles_background_variation() -> None:
    gradient = np.linspace(225, 255, LAYOUT.raster_width_px, dtype=np.uint8)
    page = np.repeat(gradient[None, :, None], LAYOUT.raster_height_px, axis=0)
    page = np.repeat(page, 3, axis=2)
    page[140, 60:150] = np.clip(page[140, 60:150].astype(np.int16) - 18, 0, 255).astype(np.uint8)
    config = _zero_clearance_config(absolute_ink_threshold=180, local_contrast_threshold=8)
    mask = detect_existing_ink(page, layout=LAYOUT, config=config)
    assert np.all(mask[140, 70:140] == 255)


def test_detection_interprets_channels_as_rgb() -> None:
    page = _white_page()
    page[80, 80] = (255, 0, 0)  # RGB red has grayscale luminance above 60.
    page[90, 90] = (0, 0, 255)  # RGB blue has grayscale luminance below 60.
    config = _zero_clearance_config(absolute_ink_threshold=60, local_contrast_threshold=255)
    mask = detect_existing_ink(page, layout=LAYOUT, config=config)
    assert mask[80, 80] == 0
    assert mask[90, 90] == 255


def test_optional_noise_filter_does_not_remove_long_thin_stroke() -> None:
    page = _white_page()
    page[40, 40] = 0
    page[60, 40:90] = 0
    config = _zero_clearance_config(min_existing_component_pixels=5)
    mask = detect_existing_ink(page, layout=LAYOUT, config=config)
    assert mask[40, 40] == 0
    assert np.all(mask[60, 40:90] == 255)


def test_mask_shapes_dtypes_binary_values_and_source_preservation() -> None:
    page = _white_page()
    page[100, 100] = (10, 20, 30)
    original = page.copy()
    masks = build_masks(page, _all_allowed(), layout=LAYOUT, config=_zero_clearance_config())
    for mask in (
        masks.existing_ink_mask,
        masks.protected_mask,
        masks.allowed_region_mask,
        masks.safe_page_interior_mask,
        masks.layout_exclusion_mask,
        masks.printable_region_mask,
    ):
        assert mask.shape == LAYOUT.image_shape
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})
    assert np.array_equal(page, original)


def test_clearance_uses_euclidean_distance_in_physical_units() -> None:
    page = _white_page()
    page[100, 100] = 0
    config = MaskingConfig(
        clearance_mm=2.0,
        printer_margin_mm=0.0,
        layout_guide_clearance_mm=0.0,
        exclude_calibration_labels=False,
    )
    masks = build_masks(page, _all_allowed(), layout=LAYOUT, config=config)
    assert masks.protected_mask[100, 102] == 255
    assert masks.protected_mask[102, 102] == 0


def test_zero_clearance_protection_equals_existing_ink_exactly() -> None:
    page = _white_page()
    page[90:93, 70:80] = 0
    masks = build_masks(page, _all_allowed(), layout=LAYOUT, config=_zero_clearance_config())
    assert np.array_equal(masks.protected_mask, masks.existing_ink_mask)


def test_repeated_mask_builds_are_deterministic() -> None:
    page = _white_page()
    cv2.circle(page, (105, 148), 20, (30, 30, 30), 2)
    config = MaskingConfig(clearance_mm=1.5)
    first = build_masks(page, _all_allowed(), layout=LAYOUT, config=config)
    second = build_masks(page, _all_allowed(), layout=LAYOUT, config=config)
    assert np.array_equal(first.existing_ink_mask, second.existing_ink_mask)
    assert np.array_equal(first.printable_region_mask, second.printable_region_mask)
