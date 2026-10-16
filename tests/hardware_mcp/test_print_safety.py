"""Artifact validation and the prepare/submit/cancel safety contract."""

from datetime import timedelta
import json

import numpy as np
from PIL import Image
import pytest

from printgpt_hardware_mcp.config import HardwareConfig
from printgpt_hardware_mcp.models import HardwareError
from printgpt_hardware_mcp.windows_printing import JobRecord

from hw_fakes import NOW, QUEUE, edit_manifest, make_service, toner_raster, write_export


def _errors(service, path):
    report = service.validate_print_artifact(str(path))
    assert not report.valid
    return " ".join(report.errors)


# --------------------------------------------------------------- validation

def test_approved_png_and_pdf_are_valid(service, export):
    png = service.validate_print_artifact(str(export["png"]))
    assert png.valid, png.errors
    assert png.page_size_mm == (210.0, 297.0) and png.pixel_size == (2480, 3508)
    pdf = service.validate_print_artifact(str(export["pdf"]))
    assert pdf.valid, pdf.errors


def test_rejects_files_outside_allowlist_and_traversal(service, config, export, tmp_path):
    outside = tmp_path / "elsewhere.png"
    Image.fromarray(toner_raster()).save(outside)
    assert "outside the allowed" in _errors(service, outside)
    sneaky = config.artifacts_root / ".." / "elsewhere.png"
    assert "outside the allowed" in _errors(service, sneaky)
    assert "does not exist" in _errors(service, config.artifacts_root / "nope.png")


def test_rejects_unsupported_formats(service, export):
    bmp = export["png"].with_suffix(".bmp")
    Image.fromarray(toner_raster()).save(bmp)
    assert "unsupported format" in _errors(service, bmp)
    assert "unsupported format" in _errors(service, export["manifest"])


def test_rejects_non_a4_pdf(service, config):
    paths = write_export(config.artifacts_root / "letter" / "run", page_mm=(215.9, 279.4))
    assert "page_size" in _errors(service, paths["pdf"])


def test_rejects_wrong_raster_dimensions(service, config):
    paths = write_export(config.artifacts_root / "small" / "run", toner_raster((2550, 3300)))
    assert "raster_size" in _errors(service, paths["png"])


def test_rejects_original_scans_and_previews(service, config, export):
    run_input = config.artifacts_root / "streamlit" / "abc123" / "input" / "upload.png"
    run_input.parent.mkdir(parents=True)
    Image.fromarray(toner_raster()).save(run_input)
    assert "never printable" in _errors(service, run_input)
    imported = config.imports_root / "run-1" / "input" / "scan.png"
    imported.parent.mkdir(parents=True)
    Image.fromarray(toner_raster()).save(imported)
    assert "never printable" in _errors(service, imported)
    # A grey scan/preview dressed up with an approval manifest is still refused.
    preview = write_export(config.artifacts_root / "preview" / "run", toner_raster(gray=True))
    assert "toner_only_pixels" in _errors(service, preview["png"])


def test_rejects_missing_and_invalid_manifests(service, config, export):
    bare = write_export(config.artifacts_root / "bare" / "run", manifest=False)
    assert "no print_approval.json" in _errors(service, bare["png"])
    export["manifest"].write_text("{not json", encoding="utf-8")
    assert "not valid JSON" in _errors(service, export["png"])


@pytest.mark.parametrize("change, expected", [
    ({"schema_version": "printgpt-mvp-run/1"}, "manifest_schema"),
    ({"approval__state": "pending"}, "explicit_approval"),
    ({"approval__approved_at": "yesterday"}, "approval_timestamp"),
    ({"validation__accepted": False}, "validation_accepted"),
    ({"final_toner_only": False}, "final_toner_only"),
    ({"contains_source_page": True}, "final_toner_only"),
    ({"requested_print_settings__copies": 2}, "requested_settings"),
    ({"input_mode": None}, "input_mode_recorded"),
])
def test_manifest_fields_are_enforced(service, export, change, expected):
    edit_manifest(export["manifest"], **change)
    assert expected in _errors(service, export["png"])


def test_rejects_simulated_validation(service, config):
    paths = write_export(config.artifacts_root / "sim" / "run", simulated=True)
    assert "validation_not_simulated" in _errors(service, paths["png"])


def test_rejects_stale_approval(service, config):
    paths = write_export(config.artifacts_root / "stale" / "run", approved_at=NOW - timedelta(hours=30))
    assert "approval_timestamp" in _errors(service, paths["png"])


def test_rejects_hash_mismatch_after_approval(service, export):
    pixels = toner_raster()
    pixels[2000:2010, 600:700] = 0
    Image.fromarray(pixels, mode="L").save(export["png"], dpi=(300, 300))
    assert "SHA-256 mismatch" in _errors(service, export["png"])


def test_rejects_ink_inside_non_printable_margin(service, config):
    pixels = toner_raster()
    pixels[5:30, 1000:1100] = 0  # 0.4-2.5 mm from the top edge; Canon margin is 5.08 mm
    paths = write_export(config.artifacts_root / "edge" / "run", pixels)
    assert "printable_area" in _errors(service, paths["png"])


def test_rejects_blank_and_full_coverage(service, config):
    blank = write_export(config.artifacts_root / "blank" / "run", np.full((3508, 2480), 255, np.uint8))
    assert "has_ink" in _errors(service, blank["png"])
    black = np.full((3508, 2480), 255, np.uint8)
    black[200:3300, 200:2280] = 0
    full = write_export(config.artifacts_root / "full" / "run", black)
    assert "plausible_coverage" in _errors(service, full["png"])


# --------------------------------------------------------------- prepare / submit

def test_prepare_returns_summary_and_does_not_print(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    assert prepared.queue == QUEUE and prepared.copies == 1 and prepared.page_count == 1
    assert prepared.settings.paper_size == "A4" and not prepared.settings.fit_to_page
    assert "Nothing has been printed" in prepared.summary
    assert printer.submitted == []
    assert service.audit.tail()[-1]["operation"] == "prepared"


def test_prepare_rejects_copies_wrong_queue_and_batches(service, export):
    with pytest.raises(HardwareError, match="exactly one copy"):
        service.prepare_print_job(str(export["png"]), copies=2)
    with pytest.raises(HardwareError, match="only the queue"):
        service.prepare_print_job(str(export["png"]), queue="Microsoft Print to PDF")
    with pytest.raises(HardwareError, match="batch"):
        service.prepare_print_job([str(export["png"]), str(export["pdf"])])


def test_prepare_rejects_invalid_artifact_and_audits(service, config):
    paths = write_export(config.artifacts_root / "sim" / "run", simulated=True)
    with pytest.raises(HardwareError, match="artifact_rejected|simulated"):
        service.prepare_print_job(str(paths["png"]))
    assert service.audit.tail()[-1]["result"] == "rejected"


def test_submit_defaults_to_dry_run(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    result = service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True)
    assert result.dry_run and result.windows_job_id is None and printer.submitted == []


def test_submit_requires_confirmation_matching_hash_and_fresh_preparation(service, clock, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    with pytest.raises(HardwareError, match="confirm_physical_print"):
        service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, False, dry_run=False)
    with pytest.raises(HardwareError, match="does not match"):
        service.submit_print_job(prepared.preparation_id, "0" * 64, True, dry_run=False)
    with pytest.raises(HardwareError, match="prepare_print_job first"):
        service.submit_print_job("made-up", prepared.artifact_sha256, True, dry_run=False)
    clock.advance(minutes=16)
    with pytest.raises(HardwareError, match="expired"):
        service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    assert printer.submitted == []


def test_submit_rechecks_hash_after_preparation(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    pixels = toner_raster()
    pixels[2500:2520, 600:800] = 0
    Image.fromarray(pixels, mode="L").save(export["png"], dpi=(300, 300))
    with pytest.raises(HardwareError, match="SHA-256"):
        service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    assert printer.submitted == []


def test_physical_submit_records_job_and_blocks_duplicates(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    result = service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    assert not result.dry_run and result.windows_job_id == 42
    assert printer.submitted == [(QUEUE, (2480, 3508), result.document_name)]
    assert result.effective_settings["paper_size"] == "A4" and result.effective_settings["duplex"] == "simplex"
    assert result.effective_settings["copies"] == 1 and result.unconfirmed_settings
    with pytest.raises(HardwareError, match="already submitted"):
        service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    with pytest.raises(HardwareError, match="already submitted"):
        service.prepare_print_job(str(export["png"]))
    entry = service.audit.tail()[-1]
    assert entry["operation"] == "submitted" and entry["windows_job_id"] == 42
    assert "error" not in entry and len(json.dumps(entry)) < 4000


def test_physical_submit_refused_when_disabled_or_not_ready(config, printer, scanner, clock, export):
    disabled = HardwareConfig(artifacts_root=config.artifacts_root, scans_dir=config.scans_dir)
    service = make_service(disabled, printer, scanner, clock)
    prepared = service.prepare_print_job(str(export["png"]))
    assert not prepared.physical_print_enabled
    with pytest.raises(HardwareError, match="disabled"):
        service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    offline = make_service(config, printer, scanner, clock, reachable=False)
    prepared = offline.prepare_print_job(str(export["png"]))
    assert not prepared.printer_ready
    with pytest.raises(HardwareError, match="not ready"):
        offline.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    assert printer.submitted == []


# --------------------------------------------------------------- cancel

def test_cancel_only_jobs_created_by_printgpt(service, printer, export):
    printer.jobs.append(JobRecord(5, "Someone else.docx", 0x08))
    with pytest.raises(HardwareError, match="only jobs submitted"):
        service.cancel_print_job(5)
    prepared = service.prepare_print_job(str(export["png"]))
    job = service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    assert service.cancel_print_job(job.windows_job_id).cancelled
    assert printer.cancelled == [job.windows_job_id] and 5 in [j.job_id for j in printer.jobs]


def test_cancel_refuses_reused_job_id(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    job = service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    printer.jobs = [JobRecord(job.windows_job_id, "Unrelated after spooler restart", 0x08)]
    with pytest.raises(HardwareError, match="ID reused"):
        service.cancel_print_job(job.windows_job_id)
    assert printer.cancelled == []


def test_job_leaving_queue_is_audited_as_completed(service, printer, export):
    prepared = service.prepare_print_job(str(export["png"]))
    service.submit_print_job(prepared.preparation_id, prepared.artifact_sha256, True, dry_run=False)
    printer.jobs.clear()
    service.list_print_jobs()
    assert service.ledger.jobs()[0]["state"] == "left_queue"
    assert service.audit.tail()[-1]["operation"] == "completed"
