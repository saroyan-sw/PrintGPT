"""Measured Canon scanner placement for emailed Scanner A4 scans. No accuracy is assumed.

A Scanner A4 scan is rotated to portrait and scaled to A4, but where the sheet
sat on the glass is unknown, so printing onto the original stays locked. One
measurement unlocks it:

1. Scan a sheet with clearly visible marks (the printed calibration page works)
   with Scan and Send, exactly as for real drawings.
2. ``locate`` lists marker corners in that scan in Scanner A4 millimetres.
3. Measure the same corners on the physical sheet with a ruler, from the paper's
   top-left corner, and write both into a measurement record. Mark at least six
   spread points ``fit`` and at least three other points ``holdout``.
4. ``fit`` estimates scan-to-paper placement from the fit points only and checks
   it on the holdout points. The profile is ``measured`` only when the holdout
   error plus the stated measurement uncertainty stays below 0.5 mm.

Record format (``printgpt-scanner-measurement/1``)::

    {"schema": "printgpt-scanner-measurement/1", "session_id": "...",
     "measurement_uncertainty_mm": 0.2,
     "points": [{"scan_mm": [x, y], "paper_mm": [x, y], "role": "fit"}, ...]}

Usage::

    python -m printgpt_mvp.scanner_profile locate <scan.pdf>
    python -m printgpt_mvp.scanner_profile fit <record.json> [--output <profile.json>]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from printgpt_alignment import DEFAULT_LAYOUT, A4CalibrationLayout
from printgpt_mvp.email_ingestion.settings import application_data_dir


SCANNER_PROFILE_ENV: Final[str] = "PRINTGPT_SCANNER_PROFILE"
RECORD_SCHEMA: Final[str] = "printgpt-scanner-measurement/1"
PROFILE_SCHEMA: Final[str] = "printgpt-scanner/1"
ACCURACY_TARGET_MM: Final[float] = 0.5
_MAX_SCALE_ERROR: Final[float] = 0.03


def scanner_profile_path() -> Path:
    configured = os.environ.get(SCANNER_PROFILE_ENV, "").strip()
    return Path(configured) if configured else application_data_dir() / "scanner_profile.json"


def _errors(matrix: np.ndarray, scan: np.ndarray, paper: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.c_[scan, np.ones(len(scan))] @ matrix[:2].T - paper, axis=1)


def _stats(errors: np.ndarray) -> dict[str, Any]:
    return {"mean_error_mm": float(errors.mean()), "max_error_mm": float(errors.max()),
            "errors_mm": [float(value) for value in errors]}


def fit_scanner(record: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Fit Scanner A4 mm -> paper mm on fit points and judge it on holdout points only."""
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"measurement record must have schema {RECORD_SCHEMA}")
    uncertainty = float(record.get("measurement_uncertainty_mm", -1))
    if not 0 < uncertainty < ACCURACY_TARGET_MM:
        raise ValueError("state measurement_uncertainty_mm (ruler reading uncertainty), above 0 and below 0.5")
    rows = record.get("points") or []
    try:
        scan = np.array([row["scan_mm"] for row in rows], dtype=float)
        paper = np.array([row["paper_mm"] for row in rows], dtype=float)
        roles = np.array([row.get("role") for row in rows])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every point needs scan_mm, paper_mm and a role") from error
    if scan.shape != (len(rows), 2) or paper.shape != scan.shape or not np.isfinite(np.r_[scan, paper]).all():
        raise ValueError("point coordinates must be finite [x, y] pairs")
    fit, holdout = roles == "fit", roles == "holdout"
    if fit.sum() < 6 or holdout.sum() < 3:
        raise ValueError("use at least six fit points and three separate holdout points")
    if np.any(np.ptp(paper[fit], axis=0) < [125, 175]):
        raise ValueError("spread the fit points across the page (at least 125 x 175 mm)")
    if set(map(tuple, paper[fit])) & set(map(tuple, paper[holdout])):
        raise ValueError("holdout points must be different locations from fit points")

    coefficients, *_ = np.linalg.lstsq(np.c_[scan[fit], np.ones(fit.sum())], paper[fit], rcond=None)
    matrix = np.eye(3)
    matrix[:2] = coefficients.T
    scales = np.linalg.norm(matrix[:2, :2], axis=0)
    if np.linalg.det(matrix[:2, :2]) <= 0 or np.any(np.abs(scales - 1.0) > _MAX_SCALE_ERROR):
        raise ValueError("implausible scan-to-paper fit: check units (mm), orientation and point pairing")
    holdout_stats = _stats(_errors(matrix, scan[holdout], paper[holdout]))
    bound = holdout_stats["max_error_mm"] + uncertainty
    measured = bound < ACCURACY_TARGET_MM
    return {
        "schema": PROFILE_SCHEMA,
        "status": "measured" if measured else "rejected",
        "measured_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "session_id": record.get("session_id"),
        "scan_to_paper_mm": matrix.tolist(),
        "offset_mm": matrix[:2, 2].tolist(),
        "axis_scale": scales.tolist(),
        "rotation_degrees": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
        "fit": _stats(_errors(matrix, scan[fit], paper[fit])),
        "holdout": holdout_stats,
        "measurement_uncertainty_mm": uncertainty,
        "accuracy_bound_mm": bound,
        "reason": None if measured else f"holdout error plus uncertainty {bound:.2f} mm is not below 0.5 mm",
        "measurement_record": record,
    }


def measured_scanner_profile(path: Path | None = None) -> dict[str, Any] | None:
    """The saved profile if it exists and passed its holdout check; otherwise ``None``."""
    path = path or scanner_profile_path()
    if not path.is_file():
        return None
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "measured":
        return None
    if not profile.get("accuracy_bound_mm", ACCURACY_TARGET_MM) < ACCURACY_TARGET_MM:
        return None
    return profile


def scan_to_paper_px(profile: dict[str, Any], layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """3x3 map from Scanner A4 canonical pixels to physical paper pixels."""
    to_mm = np.diag([layout.page_width_mm / layout.raster_width_px, layout.page_height_mm / layout.raster_height_px, 1.0])
    return np.linalg.inv(to_mm) @ np.asarray(profile["scan_to_paper_mm"], dtype=float) @ to_mm


def apply_scanner_profile(canonical_rgb: np.ndarray, profile: dict[str, Any],
                          layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> np.ndarray:
    """Resample the scan into physical paper coordinates; uncovered border is white paper."""
    matrix = scan_to_paper_px(profile, layout)
    return cv2.warpAffine(canonical_rgb, matrix[:2], (layout.raster_width_px, layout.raster_height_px),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def locate_markers(canonical_rgb: np.ndarray, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> list[dict[str, Any]]:
    """ArUco marker corners in Scanner A4 millimetres, to pair with ruler measurements."""
    from printgpt_alignment.calibration_page import aruco_dictionary

    gray = cv2.cvtColor(canonical_rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(aruco_dictionary(layout), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    scale = np.array([layout.page_width_mm / layout.raster_width_px, layout.page_height_mm / layout.raster_height_px])
    found = []
    for marker_corners, marker_id in zip(corners or (), [] if ids is None else ids.ravel()):
        for name, point in zip(("top_left", "top_right", "bottom_right", "bottom_left"), marker_corners.reshape(4, 2)):
            found.append({"marker": int(marker_id), "corner": name, "scan_mm": [round(float(v), 3) for v in point * scale]})
    return sorted(found, key=lambda item: (item["marker"], item["corner"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    locate = commands.add_parser("locate", help="list marker corners in a Scanner A4 scan (mm)")
    locate.add_argument("scan", type=Path)
    fit = commands.add_parser("fit", help="fit and holdout-check a measurement record")
    fit.add_argument("record", type=Path)
    fit.add_argument("--output", type=Path, default=None, help=f"default: {scanner_profile_path()}")
    args = parser.parse_args(argv)
    if args.command == "locate":
        from printgpt_mvp.input_normalization import InputMode, normalize_document_input

        page = normalize_document_input(args.scan, input_mode=InputMode.SCANNER_A4).canonical_rgb
        print(json.dumps(locate_markers(page), indent=2))
        return 0
    profile = fit_scanner(json.loads(args.record.read_text(encoding="utf-8")))
    output = args.output or scanner_profile_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"{profile['status']}: holdout max {profile['holdout']['max_error_mm']:.2f} mm + uncertainty "
          f"{profile['measurement_uncertainty_mm']:.2f} mm -> {output}")
    return 0 if profile["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
