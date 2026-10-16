from dataclasses import replace
import json

import cv2
import numpy as np
import pypdfium2 as pdfium
import pytest

from printgpt_alignment.calibration_page import render_calibration_image
from printgpt_alignment.geometry import A4CalibrationLayout, mm_to_px
from printgpt_generation.models import AttemptStatus, RunStatus
from printgpt_mvp import (
    InputMode,
    approve_and_export,
    canonicalize_document,
    rectangular_allowed_region,
    run_offline_document,
)


LAYOUT = A4CalibrationLayout(dpi=75.0)


def _px(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, LAYOUT.dpi)))


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("mvp")
    page = render_calibration_image(LAYOUT)
    cv2.ellipse(page, (_px(80), _px(120)), (_px(30), _px(22)), 0, 20, 300, 20, 2)
    cv2.line(page, (_px(60), _px(150)), (_px(95), _px(150)), 20, 2)
    input_path = root / "unfinished.png"
    assert cv2.imwrite(str(input_path), page)
    allowed = rectangular_allowed_region(
        x_mm=40,
        y_mm=60,
        width_mm=130,
        height_mm=180,
        layout=LAYOUT,
    )
    return run_offline_document(
        input_path,
        allowed,
        root / "run",
        seed=7,
        layout=LAYOUT,
        input_mode=InputMode.MARKED_PAGE,
        normalization_confirmed=True,
    )


def test_offline_end_to_end_uses_real_masks_validation_and_sanitized_previews(offline_run) -> None:
    run = offline_run
    assert run.generation.outcome.status is RunStatus.CANDIDATES_READY
    assert not run.generation.simulated_validation
    assert len(run.generation.candidate_ids) == 2
    assert len(run.candidates) == 2
    assert run.alignment["accepted"] is True
    assert run.canonical_page_rgb.shape == (*LAYOUT.image_shape, 3)
    for candidate in run.candidates:
        attempt = candidate.attempt
        assert attempt.validation is not None and attempt.validation.accepted
        assert attempt.preview_rgb is not None and attempt.sanitized_new_ink_mask is not None
        expected = run.canonical_page_rgb.copy()
        expected[attempt.sanitized_new_ink_mask == 255] = 0
        assert np.array_equal(attempt.preview_rgb, expected)


def test_candidate_selection_uses_ranked_validated_attempts(offline_run) -> None:
    run = offline_run
    selected = run.suggested_candidate
    assert selected is not None
    assert selected.rank == 1
    assert selected.candidate_id == "candidate-2"
    assert selected.attempt_id == "candidate-2/attempt-1"
    assert run.generation.attempt("candidate-2/attempt-0").status is AttemptStatus.VALIDATION_REJECTED
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_candidate"]["attempt_id"] == selected.attempt_id
    assert manifest["validation"]["accepted"] is True
    assert manifest["approved"] is False
    assert manifest["exported_paths"] == {"a4_pdf": None, "toner_png": None}
    assert manifest["input_normalization"]["input_mode"] == "marked_page"
    assert manifest["input_normalization"]["marker_calibration_used"] is True
    assert len(manifest["input_normalization"]["applied_transform"]["matrix"]) == 3


def test_approval_gate_rejects_missing_approval_and_unranked_attempt(offline_run) -> None:
    selected = offline_run.suggested_candidate
    assert selected is not None
    with pytest.raises(PermissionError, match="approval"):
        approve_and_export(offline_run, selected.attempt_id, approved=False)
    with pytest.raises(ValueError, match="ranked"):
        approve_and_export(offline_run, "candidate-2/attempt-0", approved=True)
    assert not (offline_run.output_dir / "print").exists()


def test_approved_output_is_toner_only_and_manifested(offline_run) -> None:
    selected = offline_run.suggested_candidate
    assert selected is not None and selected.attempt.sanitized_new_ink_mask is not None
    exported = approve_and_export(offline_run, selected.attempt_id, approved=True)
    toner = cv2.imread(str(exported.toner_png_path), cv2.IMREAD_GRAYSCALE)
    mask = selected.attempt.sanitized_new_ink_mask
    assert toner is not None and toner.shape == LAYOUT.image_shape
    assert np.all(toner[mask == 255] == 0)
    assert np.all(toner[mask == 0] == 255)
    manifest = json.loads(exported.manifest_path.read_text(encoding="utf-8"))
    assert manifest["approved"] is True
    assert manifest["selected_candidate"]["attempt_id"] == selected.attempt_id
    assert manifest["exported_paths"] == {
        "a4_pdf": "print/approved_a4_actual_size.pdf",
        "toner_png": "print/approved_toner_only.png",
    }
    assert "100%" in manifest["print_instruction"]


def test_pdf_is_exact_a4_and_incorrect_mask_dimensions_are_rejected(offline_run) -> None:
    selected = offline_run.suggested_candidate
    assert selected is not None
    exported = approve_and_export(offline_run, selected.attempt_id, approved=True)
    document = pdfium.PdfDocument(str(exported.a4_pdf_path))
    try:
        width_pt, height_pt = document[0].get_size()
    finally:
        document.close()
    assert width_pt == pytest.approx(210.0 * 72.0 / 25.4, abs=0.01)
    assert height_pt == pytest.approx(297.0 * 72.0 / 25.4, abs=0.01)

    bad_attempt = replace(
        selected.attempt,
        sanitized_new_ink_mask=np.zeros((10, 10), dtype=np.uint8),
    )
    bad_attempts = tuple(
        bad_attempt if item.attempt_id == selected.attempt_id else item
        for item in offline_run.generation.attempts
    )
    bad_generation = replace(offline_run.generation, attempts=bad_attempts)
    bad_run = replace(offline_run, generation=bad_generation)
    with pytest.raises(ValueError, match="shape"):
        approve_and_export(bad_run, selected.attempt_id, approved=True)


def test_mvp_canonicalizes_unmarked_full_frame_scan(tmp_path) -> None:
    page = np.full((*LAYOUT.image_shape, 3), 255, dtype=np.uint8)
    cv2.putText(
        page,
        "unfinished sketch",
        (60, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        2,
    )
    cv2.circle(page, (_px(105), _px(150)), _px(25), (0, 0, 0), 3)
    input_path = tmp_path / "ordinary_scan.png"
    assert cv2.imwrite(str(input_path), page)

    canonical, report = canonicalize_document(
        input_path, layout=LAYOUT, input_mode=InputMode.SCANNER_A4
    )

    assert canonical.shape == (*LAYOUT.image_shape, 3)
    assert report["accepted"] is True
    assert report["input_mode"] == "scanner_a4"
    assert report["marker_calibration_used"] is False


def test_mvp_non_a4_scan_is_rejected_without_calibrated_coordinates(tmp_path) -> None:
    input_path = tmp_path / "bad_scan.png"
    assert cv2.imwrite(str(input_path), np.full((400, 400, 3), 245, dtype=np.uint8))
    with pytest.raises(ValueError, match="No calibrated physical coordinates"):
        canonicalize_document(input_path, layout=LAYOUT, input_mode=InputMode.SCANNER_A4)


def test_non_a4_scan_does_not_run_generation(tmp_path):
    page = np.full((1000, 802, 3), 255, dtype=np.uint8)
    cv2.putText(page, "Scanner export", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.rectangle(page, (70, 260), (700, 800), (0, 0, 0), 3)
    path = tmp_path / "scan.png"
    assert cv2.imwrite(str(path), page)
    allowed = np.full(LAYOUT.image_shape, 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="recognizable full-page A4"):
        run_offline_document(
            path,
            allowed,
            tmp_path / "run",
            layout=LAYOUT,
            input_mode=InputMode.SCANNER_A4,
            normalization_confirmed=True,
        )
    assert np.all(allowed == 255)


def test_generation_requires_normalized_preview_confirmation(tmp_path):
    page = render_calibration_image(LAYOUT)
    path = tmp_path / "marked.png"
    assert cv2.imwrite(str(path), page)
    allowed = np.full(LAYOUT.image_shape, 255, dtype=np.uint8)
    with pytest.raises(PermissionError, match="preview confirmation"):
        run_offline_document(
            path,
            allowed,
            tmp_path / "run",
            layout=LAYOUT,
            input_mode=InputMode.MARKED_PAGE,
            normalization_confirmed=False,
        )
