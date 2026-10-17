"""Scanner A4 pages: preview only, locked until measured, then printable onto the original."""

import cv2
import numpy as np
import pytest

from printgpt_mvp import InputMode, approve_and_export, normalize_document_input, rectangular_allowed_region
from printgpt_mvp.pipeline import run_offline_document
from printgpt_mvp.scanner_profile import RECORD_SCHEMA, fit_scanner


def _profile(offset=(1.0, 0.5)):
    points = []
    for role, locations in (("fit", [(15, 15), (195, 15), (15, 282), (195, 282), (105, 150), (60, 220)]),
                            ("holdout", [(40, 40), (170, 260), (100, 200)])):
        for x, y in locations:
            points.append({"scan_mm": [x - offset[0], y - offset[1]], "paper_mm": [x, y], "role": role})
    return fit_scanner({"schema": RECORD_SCHEMA, "measurement_uncertainty_mm": 0.2, "points": points})


@pytest.fixture(scope="module")
def scan(tmp_path_factory):
    root = tmp_path_factory.mktemp("scanner")
    page = np.full((3508, 2480, 3), 255, np.uint8)
    cv2.ellipse(page, (900, 1400), (350, 260), 0, 20, 300, (20, 20, 20), 8)
    cv2.line(page, (600, 1750), (1100, 1750), (20, 20, 20), 8)
    path = root / "scan.png"
    assert cv2.imwrite(str(path), page)
    return path, normalize_document_input(path, input_mode=InputMode.SCANNER_A4)


def _run(scan, output, **kwargs):
    path, normalization = scan
    return run_offline_document(
        path, rectangular_allowed_region(x_mm=40, y_mm=60, width_mm=130, height_mm=180), output,
        input_mode=InputMode.SCANNER_A4, normalization_result=normalization, normalization_confirmed=True, **kwargs,
    )


def _export(run):
    return approve_and_export(run, run.suggested_candidate.attempt_id, approved=True)


def test_unconfirmed_scan_stays_preview_only(scan, tmp_path) -> None:
    run = _run(scan, tmp_path / "run")
    registration = run.alignment["registration"]
    assert registration["status"] == "preview_only" and not registration["export_allowed"]
    with pytest.raises(ValueError, match="preview only"):
        _export(run)


def test_confirmed_scan_without_measurement_is_locked(scan, tmp_path) -> None:
    run = _run(scan, tmp_path / "run", scanner_full_sheet_confirmed=True)
    registration = run.alignment["registration"]
    assert registration["status"] == "awaiting_scanner_measurement"
    assert not registration["export_allowed"]
    assert run.candidates  # generation and preview still work
    with pytest.raises(ValueError):
        _export(run)
    assert not (tmp_path / "run" / "print").exists()


def test_confirmed_scan_with_measured_profile_exports_in_paper_coordinates(scan, tmp_path) -> None:
    _, normalization = scan
    profile = _profile(offset=(1.0, 0.5))
    run = _run(scan, tmp_path / "run", scanner_full_sheet_confirmed=True, scanner_profile=profile)
    registration = run.alignment["registration"]
    assert registration["status"] == "registered" and registration["export_allowed"]
    assert registration["measured_source_accuracy_mm"] == pytest.approx(0.2, abs=1e-6)
    # The scan is resampled into paper coordinates: content moves by the measured offset.
    source_ink = np.nonzero(normalization.canonical_rgb[:, :, 0] < 128)
    paper_ink = np.nonzero(run.canonical_page_rgb[:, :, 0] < 128)
    px_per_mm = 300 / 25.4
    assert (paper_ink[1].mean() - source_ink[1].mean()) / px_per_mm == pytest.approx(1.0, abs=0.1)
    assert (paper_ink[0].mean() - source_ink[0].mean()) / px_per_mm == pytest.approx(0.5, abs=0.1)
    exported = _export(run)
    assert exported.toner_png_path.is_file()


def test_unmeasured_profile_is_refused(scan, tmp_path) -> None:
    with pytest.raises(ValueError, match="measured"):
        _run(scan, tmp_path / "run", scanner_full_sheet_confirmed=True, scanner_profile={"status": "rejected"})


def test_spare_sheet_test_export_is_allowed_and_marked(scan, tmp_path) -> None:
    import json

    run = _run(scan, tmp_path / "run", scanner_full_sheet_confirmed=True)
    exported = approve_and_export(run, run.suggested_candidate.attempt_id, approved=True, spare_sheet_test=True)
    assert exported.spare_sheet_test and exported.toner_png_path.is_file()
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["registration"]["status"] == "unregistered_spare_sheet_test"
    approval = json.loads((tmp_path / "run" / "print" / "print_approval.json").read_text(encoding="utf-8"))
    assert approval["input_mode"]["registration_status"] == "unregistered_spare_sheet_test"


def test_loop_calibration_registers_and_corrects_the_export(scan, tmp_path) -> None:
    import json

    loop_correction = [[1.0, 0.0, -0.5], [0.0, 1.0, 1.2], [0.0, 0.0, 1.0]]
    profile = {"status": "measured", "printer_queue": "Canon", "tray": "bypass", "measured_at": "2026-10-17T18:00:00",
               "accuracy_bound_mm": 0.3, "rounds": 3, "correction_mm": loop_correction}
    run = _run(scan, tmp_path / "run", scanner_full_sheet_confirmed=True, loop_profile=profile)
    registration = run.alignment["registration"]
    assert registration["status"] == "registered" and registration["export_allowed"]
    assert "loop calibration" in registration["evidence"]
    exported = _export(run)
    toner = cv2.imread(str(exported.toner_png_path), cv2.IMREAD_GRAYSCALE) < 128
    mask = run.generation.attempt(run.suggested_candidate.attempt_id).sanitized_new_ink_mask > 0
    px_per_mm = 300 / 25.4
    shift_x = (np.nonzero(toner)[1].mean() - np.nonzero(mask)[1].mean()) / px_per_mm
    shift_y = (np.nonzero(toner)[0].mean() - np.nonzero(mask)[0].mean()) / px_per_mm
    assert shift_x == pytest.approx(-0.5, abs=0.1) and shift_y == pytest.approx(1.2, abs=0.1)
    approval = json.loads((tmp_path / "run" / "print" / "print_approval.json").read_text(encoding="utf-8"))
    assert approval["printer_correction_applied"] is True
