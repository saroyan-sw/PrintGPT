"""Create reproducible Scanner A4 and Marked-page verification artifacts."""

from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import cv2

from printgpt_alignment import DEFAULT_LAYOUT, mm_to_px
from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_mvp import InputMode, normalize_document_input


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_rgb(path: Path, image) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {path}")


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("scanner_pdf", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/input_mode_verification"),
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scanner = normalize_document_input(args.scanner_pdf, input_mode=InputMode.SCANNER_A4)
    scanner_png = args.output_dir / "scanner_a4_corrected.png"
    _write_rgb(scanner_png, scanner.canonical_rgb)
    _write_json(args.output_dir / "scanner_a4_metadata.json", scanner.to_metadata())

    marked_input = render_calibration_image(DEFAULT_LAYOUT)
    px = lambda value: int(round(mm_to_px(value, DEFAULT_LAYOUT.dpi)))
    cv2.ellipse(marked_input, (px(105), px(145)), (px(30), px(22)), 0, 15, 280, 0, px(0.5))
    marked_input_path = args.output_dir / "marked_page_input.png"
    if not cv2.imwrite(str(marked_input_path), marked_input):
        raise OSError(f"could not write {marked_input_path}")
    marked = normalize_document_input(marked_input_path, input_mode=InputMode.MARKED_PAGE)
    marked_png = args.output_dir / "marked_page_canonical.png"
    _write_rgb(marked_png, marked.canonical_rgb)
    _write_json(args.output_dir / "marked_page_metadata.json", marked.to_metadata())

    _write_json(
        args.output_dir / "verification.json",
        {
            "scanner_sample": str(args.scanner_pdf),
            "scanner_auto_rotation": scanner.selected_rotation.value,
            "scanner_canonical_dimensions_px": list(scanner.canonical_dimensions),
            "scanner_crop": scanner.applied_transform.crop,
            "scanner_padding_ltrb_px": list(scanner.applied_transform.padding_ltrb),
            "marked_marker_ids": list(marked.marker_ids),
            "marked_confidence": marked.confidence,
            "marked_calibration_used": marked.marker_calibration_used,
            "artifacts": [str(scanner_png), str(marked_input_path), str(marked_png)],
        },
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
