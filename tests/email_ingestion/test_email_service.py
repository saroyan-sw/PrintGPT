import logging

import pytest

from printgpt_mvp.email_ingestion.models import ImportOutcome, PollOutcome, ScanSessionStatus
from printgpt_mvp.email_ingestion.service import UnknownSessionError
from printgpt_mvp.email_ingestion.store import IngestionStore, StoreLockTimeoutError

from email_fakes import SENDER, FakeGmail, a4_pdf_bytes, a4_png_bytes, make_service


def test_no_scanner_email_keeps_waiting(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    session = service.start_scan_session()
    result = service.poll_scan_session(session.session_id)
    assert result.outcome is PollOutcome.WAITING
    assert result.session.status is ScanSessionStatus.WAITING_FOR_EMAIL
    assert gmail.queries == [f"from:{SENDER} to:scan-inbox@example.com has:attachment newer_than:7d"]


def test_newest_scan_is_imported_automatically(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("older", minutes=0)
    gmail.add_scan("newer", minutes=5)
    session = service.start_scan_session()
    result = service.poll_scan_session(session.session_id)
    assert result.outcome is PollOutcome.IMPORTED
    imported = result.session
    assert imported.message_id == "newer"
    assert imported.status is ScanSessionStatus.IMPORTED
    assert imported.imported_path is not None and imported.imported_path.read_bytes() == a4_pdf_bytes()
    assert imported.imported_path.is_relative_to(tmp_path / "artifacts" / "email_scans")
    assert imported.sha256 and imported.run_id == f"email-{imported.sha256[:12]}"
    assert {"email_found", "downloading", "downloaded", "validating", "imported"} <= set(imported.timestamps)
    assert gmail.attachment_downloads == [("newer", "newer-att0")]


def test_unrelated_messages_are_ignored(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("scan", minutes=0)
    gmail.add_scan("other-sender", minutes=10, sender="someone@example.com")
    gmail.add_scan("other-recipient", minutes=11, recipient="else@example.com")
    gmail.add_message("no-attachment", attachments=[])
    session = service.start_scan_session()
    result = service.poll_scan_session(session.session_id)
    assert result.session.message_id == "scan"
    assert gmail.attachment_downloads == [("scan", "scan-att0")]


def test_latest_already_processed_waits_for_a_new_scan(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("first", minutes=0)
    first = service.start_scan_session()
    assert service.poll_scan_session(first.session_id).outcome is PollOutcome.IMPORTED

    second = service.start_scan_session()
    waiting = service.poll_scan_session(second.session_id)
    assert waiting.outcome is PollOutcome.WAITING
    assert "already imported" in waiting.detail
    assert len(gmail.attachment_downloads) == 1

    gmail.add_scan("second", minutes=3, pdf=False)
    imported = service.poll_scan_session(second.session_id)
    assert imported.outcome is PollOutcome.IMPORTED
    assert imported.session.message_id == "second"


def test_same_message_cannot_be_imported_twice(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("scan")
    service.poll_scan_session(service.start_scan_session().session_id)
    again = service.start_scan_session()
    result = service.import_selected_scan(again.session_id, "scan", "scan-att0")
    assert result.outcome is ImportOutcome.ALREADY_PROCESSED
    assert len(gmail.attachment_downloads) == 1


def test_identical_file_in_new_message_is_already_processed(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("original", minutes=0)
    service.poll_scan_session(service.start_scan_session().session_id)
    gmail.add_scan("forwarded-copy", minutes=2)
    session = service.start_scan_session()
    result = service.poll_scan_session(session.session_id)
    assert result.outcome is PollOutcome.WAITING
    assert "identical file" in result.detail
    # The copy is remembered, so later polls do not download it again.
    service.poll_scan_session(session.session_id)
    assert gmail.attachment_downloads.count(("forwarded-copy", "forwarded-copy-att0")) == 1


def test_pdf_is_preferred_over_image_alternative(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_message(
        "both", attachments=[("page.png", "image/png", a4_png_bytes()), ("page.pdf", "application/pdf", a4_pdf_bytes())]
    )
    result = service.poll_scan_session(service.start_scan_session().session_id)
    assert result.outcome is PollOutcome.IMPORTED
    assert result.session.original_filename == "page.pdf"


def test_multiple_candidates_require_selection(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_message(
        "two-pages",
        attachments=[("a.pdf", "application/pdf", a4_pdf_bytes()), ("b.pdf", "application/pdf", a4_pdf_bytes() + b"\n")],
    )
    session = service.start_scan_session()
    result = service.poll_scan_session(session.session_id)
    assert result.outcome is PollOutcome.NEEDS_SELECTION
    assert [item.filename for item in result.session.selection_candidates] == ["a.pdf", "b.pdf"]
    assert gmail.attachment_downloads == []
    chosen = service.import_selected_scan(session.session_id, "two-pages", "two-pages-att1")
    assert chosen.outcome is ImportOutcome.IMPORTED
    assert chosen.session.original_filename == "b.pdf"


def test_timeout_and_cancellation(tmp_path) -> None:
    service, _, clock = make_service(tmp_path)
    timed = service.start_scan_session()
    clock.advance(181)
    assert service.poll_scan_session(timed.session_id).outcome is PollOutcome.TIMED_OUT
    cancelled = service.start_scan_session()
    assert service.cancel_scan_session(cancelled.session_id).status is ScanSessionStatus.CANCELLED
    assert service.poll_scan_session(cancelled.session_id).outcome is PollOutcome.CANCELLED


def test_oversized_attachment_fails_and_does_not_block_next_session(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path, max_attachment_mb=0.01)
    gmail.add_scan("huge")
    result = service.poll_scan_session(service.start_scan_session().session_id)
    assert result.outcome is PollOutcome.FAILED
    assert "above the limit" in result.detail
    assert gmail.attachment_downloads == []
    assert service.poll_scan_session(service.start_scan_session().session_id).outcome is PollOutcome.WAITING


def test_invalid_content_is_rejected(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_message("fake-pdf", attachments=[("scan.pdf", "application/pdf", b"PK\x03\x04zipped")])
    result = service.poll_scan_session(service.start_scan_session().session_id)
    assert result.outcome is PollOutcome.FAILED
    assert result.session.status is ScanSessionStatus.FAILED
    assert not list((tmp_path / "artifacts").rglob("*.pdf"))


def test_gmail_error_keeps_session_waiting(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.fail_listing = True
    result = service.poll_scan_session(service.start_scan_session().session_id)
    assert result.outcome is PollOutcome.WAITING
    assert "503" in result.detail


def test_session_state_survives_a_new_service_instance(tmp_path) -> None:
    service, gmail, clock = make_service(tmp_path)
    session = service.start_scan_session()
    reloaded, _, _ = make_service(tmp_path, gmail=gmail, clock=clock)
    assert reloaded.get_scan_session_status(session.session_id) == session
    with pytest.raises(UnknownSessionError):
        reloaded.get_scan_session_status("scan-missing")


def test_imported_scan_is_handed_to_scanner_a4_normalization(tmp_path) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("scan")
    session = service.start_scan_session()
    service.poll_scan_session(session.session_id)
    ready, normalization = service.normalize_imported_scan(session.session_id)
    assert ready.status is ScanSessionStatus.AWAITING_CONFIRMATION
    assert normalization.canonical_rgb.shape == (3508, 2480, 3)
    assert normalization.input_mode.value == "scanner_a4"
    assert normalization.marker_calibration_used is False


def test_logs_exclude_subjects_and_bodies(tmp_path, caplog) -> None:
    service, gmail, _ = make_service(tmp_path)
    gmail.add_scan("scan", subject="Private subject line")
    gmail.add_scan("unrelated", minutes=9, sender="friend@example.com", subject="Dinner plans")
    with caplog.at_level(logging.DEBUG, logger="printgpt.email_ingestion"):
        service.poll_scan_session(service.start_scan_session().session_id)
    text = caplog.text
    assert "scan" in text and "sha256" in text
    for secret in ("Private subject line", "Dinner plans", "friend@example.com", "UHJpdmF0ZSBib2R5"):
        assert secret not in text


def test_store_lock_is_exclusive(tmp_path) -> None:
    first = IngestionStore(tmp_path / "state")
    second = IngestionStore(tmp_path / "state", lock_timeout_s=0.1)
    with first.locked():
        with pytest.raises(StoreLockTimeoutError):
            with second.locked():
                pass
    with second.locked():
        pass


def test_fake_gmail_never_used_for_real_inbox() -> None:
    assert FakeGmail().profile_email().endswith("@example.com")
