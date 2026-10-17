"""Measured scanner placement: fit, holdout check, loading, resampling, marker location."""

import json

import cv2
import numpy as np
import pytest

from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_mvp.scanner_profile import (
    RECORD_SCHEMA,
    apply_scanner_profile,
    fit_scanner,
    locate_markers,
    measured_scanner_profile,
)

LAYOUT = A4CalibrationLayout(dpi=75.0)
FIT = [(15, 15), (195, 15), (15, 282), (195, 282), (105, 150), (60, 220), (150, 80)]
HOLDOUT = [(40, 40), (170, 260), (100, 200)]


def _record(offset=(1.2, -0.8), scale=1.002, noise_mm=0.0, uncertainty=0.2, seed=0):
    rng = np.random.default_rng(seed)
    points = []
    for role, locations in (("fit", FIT), ("holdout", HOLDOUT)):
        for paper in locations:
            paper = np.array(paper, float)
            scan = (paper - offset) / scale + rng.normal(0, noise_mm, 2)
            points.append({"scan_mm": scan.tolist(), "paper_mm": paper.tolist(), "role": role})
    return {"schema": RECORD_SCHEMA, "session_id": "t", "measurement_uncertainty_mm": uncertainty, "points": points}


def test_clean_measurement_is_measured_and_recovers_the_offset() -> None:
    profile = fit_scanner(_record())
    assert profile["status"] == "measured"
    assert profile["offset_mm"] == pytest.approx([1.2, -0.8], abs=1e-6)
    assert profile["axis_scale"] == pytest.approx([1.002, 1.002], abs=1e-6)
    assert profile["accuracy_bound_mm"] == pytest.approx(0.2, abs=1e-6)


def test_noisy_holdout_is_rejected() -> None:
    profile = fit_scanner(_record(noise_mm=0.6, seed=3))
    assert profile["status"] == "rejected"
    assert "not below 0.5 mm" in profile["reason"]


@pytest.mark.parametrize(
    "broken",
    [
        lambda r: r.update(schema="other"),
        lambda r: r.update(measurement_uncertainty_mm=0),
        lambda r: r.update(points=r["points"][:5]),
        lambda r: [p.update(role="fit") for p in r["points"]],
        lambda r: [p.update(paper_mm=[p["paper_mm"][0] / 10, p["paper_mm"][1] / 10]) for p in r["points"]],
    ],
    ids=["schema", "uncertainty", "too-few", "no-holdout", "wrong-units"],
)
def test_unusable_records_are_refused(broken) -> None:
    record = _record()
    broken(record)
    with pytest.raises(ValueError):
        fit_scanner(record)


def test_only_measured_profiles_are_loaded(tmp_path) -> None:
    path = tmp_path / "scanner.json"
    assert measured_scanner_profile(path) is None
    path.write_text(json.dumps(fit_scanner(_record(noise_mm=0.6, seed=3))), encoding="utf-8")
    assert measured_scanner_profile(path) is None
    path.write_text(json.dumps(fit_scanner(_record())), encoding="utf-8")
    assert measured_scanner_profile(path)["status"] == "measured"


def test_applying_the_profile_moves_scan_content_to_paper_position() -> None:
    profile = fit_scanner(_record(offset=(3.0, 2.0), scale=1.0))
    page = np.full((*LAYOUT.image_shape, 3), 255, np.uint8)
    px = LAYOUT.dpi / 25.4
    cv2.circle(page, (int(100 * px), int(100 * px)), 4, (0, 0, 0), -1)
    moved = apply_scanner_profile(page, profile, LAYOUT)
    ys, xs = np.nonzero(moved[:, :, 0] < 128)
    assert xs.mean() / px == pytest.approx(103.0, abs=0.4)
    assert ys.mean() / px == pytest.approx(102.0, abs=0.4)


def test_markers_are_located_in_millimetres() -> None:
    layout = A4CalibrationLayout()
    found = locate_markers(cv2.cvtColor(render_calibration_image(layout), cv2.COLOR_GRAY2RGB), layout)
    assert len({item["marker"] for item in found}) == 4 and len(found) == 16
    assert all(0 <= item["scan_mm"][0] <= 210 and 0 <= item["scan_mm"][1] <= 297 for item in found)
