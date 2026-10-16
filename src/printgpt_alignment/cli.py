"""Command-line interface for calibration, rectification, and benchmarking."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json

import cv2

from .benchmark import run_benchmark
from .calibration_page import write_calibration_pdf, write_calibration_png
from .detection import detect_and_rectify_document
from .document import load_document_page
from .geometry import DEFAULT_LAYOUT
from .synthetic import generate_synthetic_benchmark
from .smartdoc import prepare_smartdoc_benchmark


def _scan_paths(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in extensions)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="printgpt-align", description="PrintGPT A4 alignment tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibration = subparsers.add_parser("calibration", help="generate print-ready calibration files")
    calibration.add_argument("--pdf", type=Path, default=Path("output/pdf/printgpt_a4_calibration_v1.pdf"))
    calibration.add_argument("--png", type=Path, default=Path("output/png/printgpt_a4_calibration_v1_300dpi.png"))

    rectify = subparsers.add_parser("rectify", help="detect and rectify one PDF page, scan, or photograph")
    rectify.add_argument("input", type=Path, help="PDF or raster image to process")
    rectify.add_argument("--page", type=int, default=1, help="1-based PDF page number (default: 1)")
    rectify.add_argument("--dpi", type=float, default=DEFAULT_LAYOUT.dpi, help="PDF rendering DPI (default: 300)")
    rectify.add_argument("--output", type=Path, required=True, help="output raster image path")
    rectify.add_argument("--report", type=Path)
    rectify.add_argument("--input-kind", choices=["auto", "scan", "photo"], default="auto")

    generate = subparsers.add_parser("generate", help="generate deterministic synthetic cases")
    generate.add_argument("--output-dir", type=Path, default=Path("artifacts/synthetic"))
    generate.add_argument("--scan-dir", type=Path, default=Path("data/scanned_docs/funsd"))
    generate.add_argument("--count", type=int, default=30)
    generate.add_argument("--seed", type=int, default=20261014)

    benchmark = subparsers.add_parser("benchmark", help="evaluate generated cases")
    benchmark.add_argument("--cases-dir", type=Path, default=Path("artifacts/synthetic"))
    benchmark.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))

    smartdoc = subparsers.add_parser("smartdoc", help="prepare SmartDoc-derived camera cases")
    smartdoc.add_argument("archive", type=Path)
    smartdoc.add_argument("--output-dir", type=Path, default=Path("artifacts/smartdoc"))
    smartdoc.add_argument("--count", type=int, default=10)

    all_command = subparsers.add_parser("all", help="generate calibration, cases, and benchmark report")
    all_command.add_argument("--cases-dir", type=Path, default=Path("artifacts/synthetic"))
    all_command.add_argument("--report-dir", type=Path, default=Path("artifacts/benchmark"))
    all_command.add_argument("--scan-dir", type=Path, default=Path("data/scanned_docs/funsd"))
    all_command.add_argument("--count", type=int, default=30)
    all_command.add_argument("--seed", type=int, default=20261014)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibration":
        print(write_calibration_pdf(args.pdf))
        print(write_calibration_png(args.png))
        return 0
    if args.command == "rectify":
        if args.page < 1:
            raise SystemExit("--page must be 1 or greater")
        try:
            image = load_document_page(
                args.input,
                page_index=args.page - 1,
                dpi=args.dpi,
            )
        except (FileNotFoundError, IndexError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        result = detect_and_rectify_document(image, input_kind=args.input_kind)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            report = result.to_dict()
            report.update(
                {
                    "input": str(args.input),
                    "input_page": args.page,
                    "pdf_render_dpi": args.dpi if args.input.suffix.lower() == ".pdf" else None,
                }
            )
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not result.accepted or result.warped is None:
            print(f"rejected: {result.reason}")
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), result.warped):
            raise SystemExit(f"could not write {args.output}")
        print(f"accepted confidence={result.confidence:.3f} orientation={result.orientation_degrees}")
        return 0
    if args.command == "generate":
        print(generate_synthetic_benchmark(args.output_dir, count=args.count, seed=args.seed, scan_paths=_scan_paths(args.scan_dir)))
        return 0
    if args.command == "benchmark":
        print(run_benchmark(args.cases_dir, args.output_dir))
        return 0
    if args.command == "smartdoc":
        print(prepare_smartdoc_benchmark(args.archive, args.output_dir, count=args.count))
        return 0
    if args.command == "all":
        print(write_calibration_pdf(Path("output/pdf/printgpt_a4_calibration_v1.pdf")))
        print(write_calibration_png(Path("output/png/printgpt_a4_calibration_v1_300dpi.png")))
        print(generate_synthetic_benchmark(args.cases_dir, count=args.count, seed=args.seed, scan_paths=_scan_paths(args.scan_dir)))
        print(run_benchmark(args.cases_dir, args.report_dir))
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
