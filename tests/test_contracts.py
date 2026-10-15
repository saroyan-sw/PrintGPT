from dataclasses import FrozenInstanceError

import pytest

from printgpt_contracts import MaskingConfig, ValidationConfig


def test_masking_config_records_effective_clearance() -> None:
    config = MaskingConfig(clearance_mm=0.75)
    assert config.to_dict()["effective_clearance_mm"] == 0.75
    with pytest.raises(FrozenInstanceError):
        config.clearance_mm = 1.0  # type: ignore[misc]


def test_zero_clearance_requires_explicit_test_mode() -> None:
    with pytest.raises(ValueError, match="zero clearance"):
        MaskingConfig(clearance_mm=0.0)
    config = MaskingConfig(clearance_mm=0.0, zero_clearance_test_mode=True)
    assert config.effective_clearance_mm == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"absolute_ink_threshold": 256}, "absolute_ink_threshold"),
        ({"printer_margin_mm": -1.0}, "printer_margin_mm"),
        ({"median_blur_px": 4}, "median_blur_px"),
    ],
)
def test_invalid_masking_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MaskingConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_final_ink_pixels": -1},
        {"max_removed_fraction": 1.1},
        {"max_final_coverage_fraction": -0.1},
        {"max_small_component_fraction": 1.01},
    ],
)
def test_invalid_validation_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ValidationConfig(**kwargs)  # type: ignore[arg-type]
