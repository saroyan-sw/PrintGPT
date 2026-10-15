"""Canonical physical and raster geometry for PrintGPT A4 pages.

The physical coordinate system is portrait A4 with its origin at the top-left
paper edge. X increases to the right and Y increases down the page. Geometry
calculations use floating-point pixel coordinates; rounding is only performed
when an integer raster dimension or pixel index is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


MM_PER_INCH: Final[float] = 25.4
A4_WIDTH_MM: Final[float] = 210.0
A4_HEIGHT_MM: Final[float] = 297.0
DEFAULT_DPI: Final[float] = 300.0


def mm_to_px(value_mm: float, dpi: float = DEFAULT_DPI) -> float:
    """Convert a physical distance in millimetres to floating-point pixels."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    return float(value_mm) * float(dpi) / MM_PER_INCH


def px_to_mm(value_px: float, dpi: float = DEFAULT_DPI) -> float:
    """Convert a pixel distance to millimetres."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    return float(value_px) * MM_PER_INCH / float(dpi)


def mm_to_pixel_index(value_mm: float, dpi: float = DEFAULT_DPI) -> int:
    """Convert millimetres to the nearest integer pixel index."""
    return int(round(mm_to_px(value_mm, dpi)))


@dataclass(frozen=True, slots=True)
class PointMM:
    """A point in the top-left-origin physical page coordinate system."""

    x: float
    y: float

    def to_pixels(self, dpi: float = DEFAULT_DPI) -> tuple[float, float]:
        return mm_to_px(self.x, dpi), mm_to_px(self.y, dpi)


@dataclass(frozen=True, slots=True)
class RectMM:
    """An axis-aligned rectangle expressed in millimetres."""

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

    @property
    def corners(self) -> tuple[PointMM, PointMM, PointMM, PointMM]:
        """Return corners in top-left, top-right, bottom-right, bottom-left order."""
        return (
            PointMM(self.x, self.y),
            PointMM(self.right, self.y),
            PointMM(self.right, self.bottom),
            PointMM(self.x, self.bottom),
        )


@dataclass(frozen=True, slots=True)
class MarkerSpec:
    """The identity and exact physical bounds of one calibration marker."""

    marker_id: int
    role: str
    bounds_mm: RectMM

    @property
    def center_mm(self) -> PointMM:
        return PointMM(
            self.bounds_mm.x + self.bounds_mm.width / 2.0,
            self.bounds_mm.y + self.bounds_mm.height / 2.0,
        )

    def corners_px(self, dpi: float = DEFAULT_DPI) -> tuple[tuple[float, float], ...]:
        return tuple(point.to_pixels(dpi) for point in self.bounds_mm.corners)


@dataclass(frozen=True, slots=True)
class A4CalibrationLayout:
    """Versioned physical specification shared by generation and detection."""

    layout_id: str = "PRINTGPT-A4-CAL-V1"
    dictionary_name: str = "DICT_4X4_50"
    page_width_mm: float = A4_WIDTH_MM
    page_height_mm: float = A4_HEIGHT_MM
    dpi: float = DEFAULT_DPI
    protected_margin_mm: float = 10.0
    marker_size_mm: float = 14.0
    marker_inset_mm: float = 12.0
    marker_quiet_zone_mm: float = 3.0

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("dpi must be positive")
        if min(self.page_width_mm, self.page_height_mm) <= 0:
            raise ValueError("page dimensions must be positive")
        if self.protected_margin_mm < 0:
            raise ValueError("protected margin cannot be negative")
        if self.marker_size_mm <= 0:
            raise ValueError("marker size must be positive")
        if self.marker_inset_mm < 0:
            raise ValueError("marker inset cannot be negative")
        if self.marker_quiet_zone_mm < 0:
            raise ValueError("marker quiet zone cannot be negative")
        if 2.0 * self.protected_margin_mm >= min(self.page_width_mm, self.page_height_mm):
            raise ValueError("protected margin leaves no usable page area")
        if self.marker_inset_mm + self.marker_size_mm > self.page_width_mm / 2.0:
            raise ValueError("horizontal marker geometry overlaps")
        if self.marker_inset_mm + self.marker_size_mm > self.page_height_mm / 2.0:
            raise ValueError("vertical marker geometry overlaps")

    @property
    def raster_width_px(self) -> int:
        return mm_to_pixel_index(self.page_width_mm, self.dpi)

    @property
    def raster_height_px(self) -> int:
        return mm_to_pixel_index(self.page_height_mm, self.dpi)

    @property
    def raster_size_px(self) -> tuple[int, int]:
        """Return raster size in image (width, height) order."""
        return self.raster_width_px, self.raster_height_px

    @property
    def image_shape(self) -> tuple[int, int]:
        """Return a grayscale array shape in NumPy (rows, columns) order."""
        return self.raster_height_px, self.raster_width_px

    @property
    def page_corners_px(self) -> tuple[tuple[float, float], ...]:
        # Pixel-edge coordinates intentionally use width/height, not last-pixel indices.
        width, height = self.raster_size_px
        return ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))

    @property
    def protected_content_mm(self) -> RectMM:
        margin = self.protected_margin_mm
        return RectMM(
            margin,
            margin,
            self.page_width_mm - 2.0 * margin,
            self.page_height_mm - 2.0 * margin,
        )

    @property
    def markers(self) -> tuple[MarkerSpec, MarkerSpec, MarkerSpec, MarkerSpec]:
        size = self.marker_size_mm
        inset = self.marker_inset_mm
        right_x = self.page_width_mm - inset - size
        bottom_y = self.page_height_mm - inset - size
        return (
            MarkerSpec(10, "top_left", RectMM(inset, inset, size, size)),
            MarkerSpec(11, "top_right", RectMM(right_x, inset, size, size)),
            MarkerSpec(12, "bottom_right", RectMM(right_x, bottom_y, size, size)),
            MarkerSpec(13, "bottom_left", RectMM(inset, bottom_y, size, size)),
        )

    @property
    def expected_marker_ids(self) -> tuple[int, int, int, int]:
        return tuple(marker.marker_id for marker in self.markers)

    def marker_by_id(self, marker_id: int) -> MarkerSpec:
        for marker in self.markers:
            if marker.marker_id == marker_id:
                return marker
        raise KeyError(f"marker ID {marker_id} is not part of {self.layout_id}")


DEFAULT_LAYOUT: Final[A4CalibrationLayout] = A4CalibrationLayout()
