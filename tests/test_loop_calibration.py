"""Print-scan loop calibration against a simulated printer + scanner (no hardware)."""

import cv2
import numpy as np
import pytest

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_mvp.loop_calibration import (
    CalibrationSession,
    load_profile,
    measure_round,
    render_target_mask,
    save_profile,
    write_target,
)

LAYOUT = A4CalibrationLayout(dpi=150.0)


def _loop(offset=(0.45, -1.2), scale=1.002, rotation_deg=0.12) -> np.ndarray:
    """True scan-after-print map in mm, like the measured Canon loop."""
    angle = np.radians(rotation_deg)
    linear = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    centre = np.array([105.0, 148.5])
    loop = np.eye(3)
    loop[:2, :2] = linear
    loop[:2, 2] = centre + np.asarray(offset) - linear @ centre
    return loop


def _print_and_scan(toner_png, loop, seed=0) -> np.ndarray:
    """Simulate printing the commanded raster and scanning the sheet."""
    toner = cv2.imread(str(toner_png), cv2.IMREAD_GRAYSCALE)
    px = np.diag([LAYOUT.raster_width_px / 210, LAYOUT.raster_height_px / 297, 1.0])
    matrix = (px @ loop @ np.linalg.inv(px))[:2]
    scan = cv2.warpAffine(toner, matrix, (LAYOUT.raster_width_px, LAYOUT.raster_height_px),
                          flags=cv2.INTER_LINEAR, borderValue=255)
    noise = np.random.default_rng(seed).normal(0, 6, scan.shape)
    scan = np.clip(cv2.GaussianBlur(scan, (0, 0), 0.8) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(scan, cv2.COLOR_GRAY2RGB)


def test_target_is_pure_toner_with_all_markers(tmp_path) -> None:
    mask = render_target_mask(LAYOUT)
    assert set(np.unique(mask)) == {0, 255}
    png, pdf = write_target(tmp_path, layout=LAYOUT)
    assert png.is_file() and pdf.is_file()
    measurement = measure_round(_print_and_scan(png, np.eye(3)), LAYOUT)
    assert measurement["markers_found"] == 24
    assert measurement["printed_error"]["max_mm"] < 0.2


def test_rounds_converge_and_the_profile_is_verified_on_a_corrected_round(tmp_path) -> None:
    loop = _loop()
    session = CalibrationSession.start(tmp_path / "sessions", "Canon Generic Plus UFR II", "bypass", )
    errors = []
    for round_index in range(3):
        pdf = session.target_for_next_round(LAYOUT)
        result = session.add_scan(_print_and_scan(pdf.with_name("approved_toner_only.png"), loop, seed=round_index),
                                  f"scan-{round_index}", LAYOUT)
        errors.append(result["printed_error"]["max_mm"])
    assert errors[0] > 1.0                  # uncorrected loop: ~1.3 mm like the real test print
    assert errors[1] < 0.3 and errors[2] < 0.3  # corrected rounds land within a fraction of a mm
    profile = session.profile()
    assert profile["verified_round"] in (2, 3) and profile["accuracy_bound_mm"] < 0.3
    # The correction inverts the loop: loop @ correction is the identity.
    assert np.allclose(loop @ np.asarray(profile["correction_mm"]), np.eye(3), atol=0.05)  # within 0.05 mm
    path = save_profile(profile, tmp_path / "profiles")
    assert path.is_file()
    assert load_profile("Canon Generic Plus UFR II", "bypass", tmp_path / "profiles")["rounds"] == 3
    assert load_profile("Other printer", "bypass", tmp_path / "profiles") is None


def test_fit_to_page_printing_is_detected_and_does_not_update_the_correction(tmp_path) -> None:
    session = CalibrationSession.start(tmp_path, "Canon", "bypass")
    pdf = session.target_for_next_round(LAYOUT)
    fitted = _loop(offset=(0.0, -1.5), scale=0.952, rotation_deg=0.0)
    result = session.add_scan(_print_and_scan(pdf.with_name("approved_toner_only.png"), fitted), "fit", LAYOUT)
    assert any("Actual size" in problem for problem in result["problems"])
    assert np.allclose(session.correction, np.eye(3))
    with pytest.raises(ValueError):
        session.profile()


def test_an_uncorrected_round_alone_cannot_be_saved(tmp_path) -> None:
    session = CalibrationSession.start(tmp_path, "Canon", "bypass")
    pdf = session.target_for_next_round(LAYOUT)
    session.add_scan(_print_and_scan(pdf.with_name("approved_toner_only.png"), np.eye(3)), "perfect", LAYOUT)
    with pytest.raises(ValueError, match="corrected round"):
        session.profile()


def test_a_page_without_enough_markers_is_refused() -> None:
    blank = np.full((*LAYOUT.image_shape, 3), 255, np.uint8)
    with pytest.raises(ValueError, match="calibration markers"):
        measure_round(blank, LAYOUT)


def test_sessions_survive_reloading(tmp_path) -> None:
    session = CalibrationSession.start(tmp_path, "Canon", "bypass")
    pdf = session.target_for_next_round(LAYOUT)
    session.add_scan(_print_and_scan(pdf.with_name("approved_toner_only.png"), _loop()), "r1", LAYOUT)
    again = CalibrationSession.load(session.directory)
    assert again.next_round == 2 and np.allclose(again.correction, session.correction)


def test_an_upside_down_scan_is_turned_and_measured(tmp_path) -> None:
    png, _ = write_target(tmp_path, layout=LAYOUT)
    scan = np.ascontiguousarray(np.rot90(_print_and_scan(png, _loop()), 2))
    measurement = measure_round(scan, LAYOUT)
    assert measurement["scan_turned_degrees"] == 180 and not measurement["problems"]
    assert measurement["offset_mm_at_centre"] == pytest.approx([0.45, -1.2], abs=0.2)
