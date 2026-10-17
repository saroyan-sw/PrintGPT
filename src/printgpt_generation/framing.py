"""Pixel windows around the printable region, used for crop framing and close-ups.

A window (``PixelBox``) is always expressed in canonical page pixels. Crop
framing sends a generator only the window around the printable region plus
enough surrounding drawing for context, grown to an aspect ratio the provider
can return. When the page itself is too small for that ratio, the remainder is
recorded as right/bottom padding so it can be cropped off again exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Axis-aligned window ``[x, x + width) x [y, y + height)`` in canonical pixels."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not all(isinstance(value, (int, np.integer)) for value in self.as_tuple()):
            raise TypeError(f"pixel box values must be integers, got {self.as_tuple()}")
        if self.x < 0 or self.y < 0 or self.width < 1 or self.height < 1:
            raise ValueError(f"invalid pixel box {self.as_tuple()}")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def fits(self, shape: tuple[int, ...]) -> bool:
        return self.right <= shape[1] and self.bottom <= shape[0]

    def is_full(self, shape: tuple[int, ...]) -> bool:
        return self.x == 0 and self.y == 0 and (self.height, self.width) == tuple(shape[:2])

    def slices(self) -> tuple[slice, slice]:
        return slice(self.y, self.bottom), slice(self.x, self.right)

    def crop(self, image: np.ndarray) -> np.ndarray:
        return image[self.slices()]


def full_box(shape: tuple[int, ...]) -> PixelBox:
    return PixelBox(0, 0, int(shape[1]), int(shape[0]))


def mask_bounds(mask: np.ndarray) -> PixelBox | None:
    """Bounding box of the non-zero pixels, or ``None`` for an empty mask."""
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return PixelBox(int(x), int(y), int(width), int(height))


def expand(box: PixelBox, margin_px: float, shape: tuple[int, ...]) -> PixelBox:
    """Grow ``box`` by ``margin_px`` on every side, clipped to the page."""
    margin = max(0, int(ceil(margin_px)))
    x0, y0 = max(0, box.x - margin), max(0, box.y - margin)
    x1, y1 = min(int(shape[1]), box.right + margin), min(int(shape[0]), box.bottom + margin)
    return PixelBox(x0, y0, x1 - x0, y1 - y0)


def parse_aspect(aspect: str) -> float:
    """Return width / height for an ``"W:H"`` string."""
    width, height = (float(part) for part in aspect.split(":"))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid aspect ratio {aspect!r}")
    return width / height


def _place(center: float, size: int, limit: int) -> int:
    """Start offset of a ``size`` span centred on ``center`` and kept inside ``[0, limit)``."""
    start = int(round(center - size / 2.0))
    return min(max(0, start), limit - size)


@dataclass(frozen=True, slots=True)
class AspectFrame:
    """A page window plus the right/bottom padding that gives it ``aspect``."""

    box: PixelBox
    aspect: str
    canvas_width: int
    canvas_height: int

    @property
    def padding(self) -> tuple[int, int]:
        return self.canvas_width - self.box.width, self.canvas_height - self.box.height


def fit_aspect(box: PixelBox, aspect: str, shape: tuple[int, ...]) -> AspectFrame:
    """Grow ``box`` on the page to ``aspect``; pad only what the page cannot supply.

    The result always contains ``box``: the grown window is centred on it and
    then shifted, never shrunk, to stay on the page.
    """
    page_height, page_width = int(shape[0]), int(shape[1])
    if not box.fits(shape):
        raise ValueError(f"box {box.as_tuple()} lies outside a {page_width}x{page_height} page")
    ratio = parse_aspect(aspect)
    if box.width / box.height < ratio:
        canvas_width, canvas_height = int(ceil(box.height * ratio)), box.height
    else:
        canvas_width, canvas_height = box.width, int(ceil(box.width / ratio))
    # Only the dimension that grew can exceed the page; the excess becomes padding.
    width, height = min(canvas_width, page_width), min(canvas_height, page_height)
    x = _place(box.x + box.width / 2.0, width, page_width)
    y = _place(box.y + box.height / 2.0, height, page_height)
    return AspectFrame(PixelBox(x, y, width, height), aspect, canvas_width, canvas_height)


def best_aspect_frame(box: PixelBox, aspects: tuple[str, ...], shape: tuple[int, ...]) -> AspectFrame:
    """The supported aspect needing the smallest canvas, preferring less padding on ties."""
    if not aspects:
        raise ValueError("at least one aspect ratio is required")
    frames = [fit_aspect(box, aspect, shape) for aspect in aspects]
    return min(
        frames,
        key=lambda frame: (frame.canvas_width * frame.canvas_height, sum(frame.padding), aspects.index(frame.aspect)),
    )
