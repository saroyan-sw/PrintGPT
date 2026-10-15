import math

import pytest

from printgpt_alignment.geometry import (
    A4CalibrationLayout,
    DEFAULT_LAYOUT,
    mm_to_pixel_index,
    mm_to_px,
    px_to_mm,
)


def test_a4_raster_dimensions_at_300_dpi() -> None:
    assert DEFAULT_LAYOUT.raster_size_px == (2480, 3508)
    assert DEFAULT_LAYOUT.image_shape == (3508, 2480)


@pytest.mark.parametrize("millimetres", [0.0, 1.0, 10.0, 14.0, 210.0, 297.0])
def test_mm_px_conversion_round_trip(millimetres: float) -> None:
    recovered = px_to_mm(mm_to_px(millimetres))
    assert math.isclose(recovered, millimetres, abs_tol=1e-12)


def test_integer_conversion_is_explicit() -> None:
    assert mm_to_pixel_index(210.0) == 2480
    assert mm_to_pixel_index(297.0) == 3508


def test_protected_content_rectangle() -> None:
    protected = DEFAULT_LAYOUT.protected_content_mm
    assert (protected.x, protected.y) == (10.0, 10.0)
    assert (protected.width, protected.height) == (190.0, 277.0)
    assert (protected.right, protected.bottom) == (200.0, 287.0)


def test_marker_identity_and_physical_positions() -> None:
    observed = [
        (marker.marker_id, marker.role, marker.bounds_mm.x, marker.bounds_mm.y)
        for marker in DEFAULT_LAYOUT.markers
    ]
    assert observed == [
        (10, "top_left", 12.0, 12.0),
        (11, "top_right", 184.0, 12.0),
        (12, "bottom_right", 184.0, 271.0),
        (13, "bottom_left", 12.0, 271.0),
    ]


def test_unknown_marker_id_is_rejected() -> None:
    with pytest.raises(KeyError):
        DEFAULT_LAYOUT.marker_by_id(49)


@pytest.mark.parametrize("dpi", [0.0, -1.0])
def test_invalid_dpi_is_rejected(dpi: float) -> None:
    with pytest.raises(ValueError, match="dpi must be positive"):
        A4CalibrationLayout(dpi=dpi)
