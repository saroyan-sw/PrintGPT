"""Measured printer affine correction. No hardware accuracy is assumed.

Run `python -m printgpt_alignment.printer_calibration --help` for the workflow.
All measurements are top-left paper millimetres, never auto-fitted scan pixels.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas


def _points(record):
    rows = record.get("points", [])
    if len(rows) < 6:
        raise ValueError("at least six measured points are required")
    try:
        expected = np.array([r["expected_mm"] for r in rows], dtype=float)
        measured = np.array([r["measured_mm"] for r in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("complete expected_mm and measured_mm for every point") from exc
    if expected.shape != (len(rows), 2) or measured.shape != expected.shape or not np.isfinite(np.r_[expected, measured]).all():
        raise ValueError("point coordinates must be finite Nx2 arrays; fill all null measurements")
    if np.any(expected < 0) or np.any(expected > [210, 297]):
        raise ValueError("expected coordinates must be inside A4")
    return rows, expected, measured


def _spread(points):
    if len(points) < 6 or np.any(np.ptp(points, axis=0) < [125, 175]) or np.linalg.matrix_rank(np.c_[points, np.ones(len(points))]) < 3:
        raise ValueError("use at least six independent points spread across the page, not a line or small patch")


def _stats(errors):
    lengths = np.linalg.norm(errors, axis=1)
    return {"mean_error_mm": float(lengths.mean()), "max_error_mm": float(lengths.max()),
            "p95_error_mm": float(np.percentile(lengths, 95)), "errors_mm": lengths.tolist()}


def correction_id(matrix):
    return sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest()[:16]


def fit_printer(record: dict) -> dict:
    """Fit commanded -> physical mm on fit points; reserve distinct holdout points."""
    rows, expected, measured = _points(record)
    if record.get("applied_correction_id") is not None:
        raise ValueError("fit requires an uncorrected print; do not fit correction twice")
    fit = np.array([r.get("role") == "fit" for r in rows])
    holdout = np.array([r.get("role") == "holdout" for r in rows])
    _spread(expected[fit]); _spread(expected[holdout])
    if set(map(tuple, expected[fit])) & set(map(tuple, expected[holdout])):
        raise ValueError("holdout locations must not be reused as fit locations")
    A = np.c_[expected[fit], np.ones(fit.sum())]
    coefficients, *_ = np.linalg.lstsq(A, measured[fit], rcond=None)
    forward = np.eye(3); forward[:2] = coefficients.T
    if not np.isfinite(forward).all() or np.linalg.det(forward[:2,:2]) <= 0 or np.linalg.cond(forward[:2,:2]) > 10:
        raise ValueError("invalid printer fit; check units, orientation and point correspondence")
    correction = np.linalg.inv(forward)
    residual = np.c_[expected, np.ones(len(expected))] @ forward[:2].T - measured
    return {"schema": "printgpt-printer/1", "status": "provisional", "measurement_source": record.get("measurement_source", "unknown"),
            "settings": record.get("settings", {}), "calibration_session": record.get("session_id"),
            "command_to_physical_mm": forward.tolist(), "correction_mm": correction.tolist(),
            "correction_id": correction_id(correction.tolist()),
            "fit": _stats(residual[fit]), "independent_holdout": _stats(residual[holdout]),
            "offset_mm": forward[:2,2].tolist(),
            "axis_scale": np.linalg.norm(forward[:2,:2], axis=0).tolist(),
            "rotation_degrees": float(np.degrees(np.arctan2(forward[1,0], forward[0,0]))),
            "axis_dot_product": float(np.dot(forward[:2,0], forward[:2,1])),
            "calibration_record": record,
            "note": "Fit residual is NOT corrected-print accuracy. Validate corrected reprints on separate sheets."}


def validate_printer(profile: dict, record: dict, *, tolerance_mm: float = .5) -> dict:
    """Evaluate corrected reprints without fitting them again (no circular score)."""
    if not np.isfinite(tolerance_mm) or not 0 < tolerance_mm <= .5:
        raise ValueError("validation tolerance must be positive and no greater than 0.5 mm")
    if record.get("session_id") in (None, profile.get("calibration_session")):
        raise ValueError("validation requires a separate measurement session")
    if record.get("settings") != profile.get("settings") or not profile.get("settings"):
        raise ValueError("printer, driver, tray, side, orientation and paper settings must match")
    required_settings = ("printer", "driver", "tray", "side", "orientation", "paper", "scaling")
    if any(not str(profile["settings"].get(key, "")).strip() for key in required_settings):
        raise ValueError("record all printer, driver, tray, side, orientation, paper and scaling settings")
    if record.get("applied_correction_id") != profile["correction_id"]:
        raise ValueError("validation must measure the target printed with this correction")
    rows, expected, measured = _points(record)
    sheets = sorted(set(str(r.get("sheet", "")) for r in rows))
    if len(sheets) < 5 or "" in sheets:
        raise ValueError("at least five separate corrected sheets are required")
    for sheet in sheets:
        subset = expected[[str(r.get("sheet")) == sheet for r in rows]]
        if len(set(map(tuple, subset))) < 9:
            raise ValueError("measure at least nine distinct locations on each validation sheet")
        _spread(subset)
    uncertainty = float(record.get("measurement_uncertainty_mm", float("nan")))
    if not np.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("record a positive measurement uncertainty in millimetres")
    margins = np.asarray(record.get("safe_margins_mm"), dtype=float)
    if margins.shape != (4,) or not np.isfinite(margins).all() or np.any(margins < 0) or np.any(margins > 30):
        raise ValueError("record measured safe margins [left, top, right, bottom] in mm")
    errors = measured - expected
    stats = _stats(errors)
    repeated = []
    common = set(map(tuple, expected))
    for sheet in sheets:
        common &= set(map(tuple, expected[[str(r.get("sheet")) == sheet for r in rows]]))
    if len(common) < 9:
        raise ValueError("repeat at least nine common locations on all five sheets")
    for point in common:
        sample = errors[np.all(expected == point, axis=1)]
        repeated.append(float(np.linalg.norm(sample[:,None] - sample[None,:], axis=2).max()))
    physical = profile.get("measurement_source") == record.get("measurement_source") == "physical"
    verified = physical and record.get("measurement_geometry_verified") is True and stats["max_error_mm"] + uncertainty < tolerance_mm
    result = dict(profile)
    result.update(status="measured" if verified else "not_verified", validation=stats,
                  measurement_uncertainty_mm=uncertainty, safe_margins_mm=margins.tolist(),
                  max_repeat_displacement_mm=max(repeated), validation_sheet_count=len(sheets),
                  validation_record=record, tolerance_mm=tolerance_mm,
                  note="Measured under recorded conditions, not a guarantee of future feed position; source-registration error is separate.")
    return result


def validated_correction(profile: dict) -> np.ndarray:
    # Recompute evidence rather than trusting a hand-edited status field.
    fitted = fit_printer(profile["calibration_record"])
    checked = validate_printer(fitted, profile["validation_record"], tolerance_mm=profile.get("tolerance_mm", .5))
    if checked["status"] != "measured":
        raise ValueError("printer correction requires physical, independently validated repeat measurements")
    if profile.get("correction_id") != fitted["correction_id"] or not np.allclose(profile["correction_mm"], fitted["correction_mm"], atol=1e-10, rtol=0):
        raise ValueError("printer correction differs from the measured fit")
    return np.asarray(fitted["correction_mm"], dtype=float)


def write_target(pdf_path: str | Path, measurements_path: str | Path, profile: dict | None = None):
    """Vector A4 target, readable centre labels, and edge visibility ladders."""
    pdf_path, measurements_path = Path(pdf_path), Path(measurements_path)
    if measurements_path.exists():
        raise FileExistsError("measurement file already exists; choose a new session path to preserve recorded data")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    measurements_path.parent.mkdir(parents=True, exist_ok=True)
    correction = np.eye(3) if profile is None else np.asarray(profile["correction_mm"], dtype=float)
    if correction.shape != (3,3) or not np.isfinite(correction).all() or not np.allclose(correction[2], [0,0,1]):
        raise ValueError("target requires a finite affine printer correction")
    pdf = canvas.Canvas(str(pdf_path), pagesize=(210*mm, 297*mm))
    pdf.setTitle("PrintGPT Canon 2425i physical registration target")
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(15*mm, 287*mm, "PrintGPT | A4 physical registration | TOP")
    pdf.setFont("Helvetica", 7)
    pdf.drawString(15*mm, 283*mm, "Actual Size / 100%. No fit, shrink, borderless expansion or auto rotation.")
    rows = []
    for j, y in enumerate(np.linspace(25, 272, 7)):
        for i, x in enumerate(np.linspace(20, 190, 5)):
            name = f"P{j*5+i+1:02d}"
            cx, cy, _ = correction @ [x, y, 1]
            pdf.setLineWidth(.15*mm)
            pdf.line((cx-2)*mm, (297-cy)*mm, (cx+2)*mm, (297-cy)*mm)
            pdf.line(cx*mm, (297-cy-2)*mm, cx*mm, (297-cy+2)*mm)
            pdf.setFont("Helvetica", 6)
            pdf.drawCentredString(cx*mm, (297-cy-4)*mm, f"{name} {x:.2f},{y:.2f}")
            rows.append({"id": name, "sheet": "sheet-1", "role": "fit" if (i+j)%2 == 0 else "holdout",
                         "expected_mm": [float(x), float(y)], "measured_mm": [None, None]})
    # Edge ladders remain uncorrected: measure driver clipping separately.
    for d in range(1, 13):
        pdf.setLineWidth(.15*mm)
        for x in (d, 210-d):
            pdf.line(x*mm, 125*mm, x*mm, 129*mm)
        for y in (d, 297-d):
            pdf.line(170*mm, y*mm, 174*mm, y*mm)
    pdf.setFont("Helvetica", 7)
    pdf.drawString(15*mm, 10*mm, "Centres are measured from physical LEFT/TOP edges. Edge ladders: 1-12 mm.")
    pdf.drawString(15*mm, 6*mm, "Target: " + ("uncorrected fit" if profile is None else "corrected verification " + profile["correction_id"]))
    pdf.showPage(); pdf.save()
    template = {"schema": "printgpt-measurements/1", "session_id": "replace-with-unique-session",
                "measurement_source": "physical", "measurement_geometry_verified": False,
                "measurement_uncertainty_mm": None, "safe_margins_mm": [None]*4,
                "settings": {"printer": "Canon imageRUNNER 2425i", "driver": "record version",
                             "tray": "record tray", "side": "simplex", "orientation": "portrait; mark feed edge",
                             "paper": "record stock and weight", "scaling": "100%; no auto rotation"},
                "applied_correction_id": None if profile is None else profile["correction_id"], "points": rows}
    if profile:
        template["settings"] = profile["settings"]
    measurements_path.write_text(json.dumps(template, indent=2)+"\n", encoding="utf-8")
    return pdf_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    target = sub.add_parser("target")
    target.add_argument("--pdf", type=Path, required=True)
    target.add_argument("--measurements", type=Path, required=True)
    target.add_argument("--profile", type=Path)
    fit = sub.add_parser("fit"); fit.add_argument("measurements", type=Path); fit.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate"); validate.add_argument("profile", type=Path); validate.add_argument("measurements", type=Path); validate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    if args.command == "target":
        print(write_target(args.pdf, args.measurements, read(args.profile) if args.profile else None))
    else:
        result = fit_printer(read(args.measurements)) if args.command == "fit" else validate_printer(read(args.profile), read(args.measurements))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(result["status"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
