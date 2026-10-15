"""Write a deterministic offline masking example and its diagnostics."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import cv2
import numpy as np

from printgpt_alignment.geometry import DEFAULT_LAYOUT, mm_to_px
from printgpt_contracts import MaskingConfig, ValidationConfig

from .core import build_masks, validate_candidate
from .diagnostics import diagnostic_images


def _pixel(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, DEFAULT_LAYOUT.dpi)))


def _write_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {path}")


def run_example(output_dir: str | Path) -> Path:
    """Create a synthetic page and return the diagnostics directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    page = np.full((*DEFAULT_LAYOUT.image_shape, 3), 255, dtype=np.uint8)
    cv2.line(page, (_pixel(60), _pixel(145)), (_pixel(105), _pixel(145)), (35, 35, 35), 5)
    cv2.ellipse(
        page,
        (_pixel(85), _pixel(125)),
        (_pixel(22), _pixel(16)),
        0,
        15,
        330,
        (110, 110, 110),
        3,
    )

    allowed = np.zeros(DEFAULT_LAYOUT.image_shape, dtype=np.uint8)
    allowed[_pixel(70) : _pixel(220), _pixel(45) : _pixel(170)] = 255
    candidate = np.zeros(DEFAULT_LAYOUT.image_shape, dtype=np.uint8)
    cv2.line(candidate, (_pixel(103), _pixel(145)), (_pixel(155), _pixel(130)), 255, 5)
    cv2.circle(candidate, (_pixel(185), _pixel(145)), _pixel(4), 255, 2)  # outside allowed

    masks = build_masks(
        page,
        allowed,
        layout=DEFAULT_LAYOUT,
        config=MaskingConfig(clearance_mm=0.5),
    )
    result = validate_candidate(
        candidate,
        masks,
        config=ValidationConfig(max_removed_fraction=0.35),
    )
    for name, image in diagnostic_images(page, candidate, masks, result).items():
        _write_rgb(output / f"{name}.png", image)
    (output / "validation.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("artifacts/masking_example"))
    args = parser.parse_args(argv)
    print(run_example(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
