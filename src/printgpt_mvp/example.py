"""Create a synthetic scan and run the complete approved offline MVP path."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import cv2

from printgpt_alignment import DEFAULT_LAYOUT, mm_to_px
from printgpt_alignment.calibration_page import render_calibration_image

from .pipeline import approve_and_export, rectangular_allowed_region, run_offline_document
from .input_normalization import InputMode


def _px(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, DEFAULT_LAYOUT.dpi)))


def write_demo_scan(path: Path) -> Path:
    """Write an A4 calibration page with a simple unfinished black drawing."""
    page = render_calibration_image(DEFAULT_LAYOUT)
    thickness = max(2, _px(0.6))
    cv2.ellipse(page, (_px(80), _px(120)), (_px(30), _px(22)), 0, 20, 300, 20, thickness)
    cv2.line(page, (_px(60), _px(150)), (_px(95), _px(150)), 20, thickness)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), page):
        raise OSError(f"failed to write {path}")
    return path


def run_example(output_dir: Path) -> Path:
    root = Path(output_dir)
    input_path = write_demo_scan(root / "input" / "unfinished_a4_scan.png")
    allowed = rectangular_allowed_region(x_mm=40, y_mm=60, width_mm=130, height_mm=180)
    run = run_offline_document(
        input_path,
        allowed,
        root / "run",
        user_prompt="",
        seed=7,
        input_mode=InputMode.MARKED_PAGE,
        normalization_confirmed=True,
    )
    selected = run.suggested_candidate
    if selected is None:
        raise RuntimeError("offline demonstration produced no valid candidate")
    return approve_and_export(run, selected.attempt_id, approved=True).a4_pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("artifacts/mvp_example"))
    args = parser.parse_args(argv)
    print(run_example(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
