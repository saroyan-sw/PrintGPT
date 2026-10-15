"""Run alignment detection against synthetic ground truth and write reports."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import cv2
import numpy as np

from .detection import QualityThresholds, detect_and_rectify
from .geometry import A4CalibrationLayout, DEFAULT_LAYOUT
from .metrics import draw_alignment_overlay, evaluate_alignment


def run_benchmark(
    cases_dir: str | Path,
    output_dir: str | Path,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    thresholds: QualityThresholds = QualityThresholds(),
) -> Path:
    cases_root = Path(cases_dir)
    output = Path(output_dir)
    overlays_dir = output / "overlays"
    rectified_dir = output / "rectified"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cases_root / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, object]] = []

    for case in manifest["cases"]:
        image_path = cases_root / case["image"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"failed to read {image_path}")
        result = detect_and_rectify(image, layout, thresholds)
        record: dict[str, object] = {
            "case_id": case["case_id"],
            "category": case["category"],
            "expected_accept": bool(case["expected_accept"]),
            "accepted": result.accepted,
            "reason": result.reason,
            "confidence": result.confidence,
            "orientation_degrees": result.orientation_degrees,
            "marker_reprojection_rmse_px": result.reprojection_rmse_px,
            "marker_reprojection_max_px": result.reprojection_max_px,
            "marker_reprojection_inlier_max_px": result.reprojection_inlier_max_px,
        }
        if result.accepted and result.source_to_canonical is not None:
            ground_truth = np.asarray(case["canonical_to_source"], dtype=np.float64)
            metrics = evaluate_alignment(ground_truth, result.source_to_canonical, layout)
            record.update(metrics.to_dict())
            overlay = draw_alignment_overlay(image, ground_truth, result.source_to_canonical, layout)
            cv2.imwrite(str(overlays_dir / f"{case['case_id']}.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if result.warped is not None:
                cv2.imwrite(str(rectified_dir / f"{case['case_id']}.jpg"), result.warped, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append(record)

    valid = [record for record in records if record["expected_accept"]]
    invalid = [record for record in records if not record["expected_accept"]]
    accepted_valid = [record for record in valid if record["accepted"]]
    false_accepts = [record for record in invalid if record["accepted"]]
    max_errors = [float(record["max_error_mm"]) for record in accepted_valid if "max_error_mm" in record]
    mean_errors = [float(record["mean_error_mm"]) for record in accepted_valid if "mean_error_mm" in record]
    summary = {
        "total_cases": len(records),
        "expected_valid_cases": len(valid),
        "expected_invalid_cases": len(invalid),
        "accepted_valid_cases": len(accepted_valid),
        "valid_failure_rate": (len(valid) - len(accepted_valid)) / len(valid) if valid else 0.0,
        "invalid_false_accept_rate": len(false_accepts) / len(invalid) if invalid else 0.0,
        "mean_alignment_error_mm": float(np.mean(mean_errors)) if mean_errors else None,
        "maximum_alignment_error_mm": max(max_errors) if max_errors else None,
        "target_max_error_mm": 0.5,
        "target_met": bool(max_errors) and max(max_errors) < 0.5,
    }
    report = {"summary": summary, "cases": records}
    report_path = output / "benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    csv_path = output / "benchmark_cases.csv"
    fieldnames = sorted({key for record in records for key in record})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return report_path
