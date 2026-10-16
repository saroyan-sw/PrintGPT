from pathlib import Path

import pypdfium2 as pdfium
import pytest

from printgpt_mvp.email_ingestion.downloader import (
    AttachmentRejectedError,
    safe_destination,
    sanitize_filename,
    save_attachment_atomically,
)
from printgpt_mvp.email_ingestion.matcher import (
    build_search_query,
    classify_attachment,
    parse_scan_message,
    preferred_attachments,
)
from printgpt_mvp.email_ingestion.models import ScanFormat

from email_fakes import RECIPIENT, SENDER, FakeGmail, a4_pdf_bytes, a4_png_bytes


def _parse(gmail: FakeGmail, message_id: str, subject_filter: str | None = None):
    return parse_scan_message(
        gmail.get_message(message_id), sender=SENDER, recipient=RECIPIENT, subject_filter=subject_filter
    )


def test_sender_must_match_exactly() -> None:
    gmail = FakeGmail()
    gmail.add_scan("ok")
    gmail.add_scan("lookalike", sender=f"x{SENDER}")
    assert _parse(gmail, "ok") is not None
    assert _parse(gmail, "lookalike") is None


def test_recipient_must_be_the_configured_inbox() -> None:
    gmail = FakeGmail()
    gmail.add_scan("other", recipient="someone@example.com")
    assert _parse(gmail, "other") is None


def test_optional_subject_filter() -> None:
    gmail = FakeGmail()
    gmail.add_scan("canon-default", subject="Scanned from a Canon")
    gmail.add_scan("printgpt", subject="PrintGPT Scan 0012")
    assert _parse(gmail, "canon-default") is not None
    assert _parse(gmail, "canon-default", "PrintGPT Scan") is None
    assert _parse(gmail, "printgpt", "printgpt scan") is not None
    assert 'subject:"PrintGPT Scan"' in build_search_query(SENDER, RECIPIENT, 7, "PrintGPT Scan")


def test_attachments_are_detected_in_nested_parts() -> None:
    gmail = FakeGmail()
    gmail.add_scan("nested")
    message = gmail.get_message("nested")
    message["payload"]["parts"] = [{"mimeType": "multipart/related", "parts": message["payload"]["parts"]}]
    parsed = _parse(gmail, "nested")
    assert parsed is not None and parsed.attachments[0].scan_format is ScanFormat.PDF


@pytest.mark.parametrize(
    ("filename", "mime", "expected"),
    [
        ("scan.pdf", "application/pdf", ScanFormat.PDF),
        ("SCAN.JPG", "image/jpeg", ScanFormat.JPEG),
        ("scan.tif", "image/tiff", ScanFormat.TIFF),
        ("scan.pdf", "application/octet-stream", ScanFormat.PDF),
        ("scan.pdf", "image/png", None),
        ("scan.zip", "application/zip", None),
        ("setup.exe", "application/octet-stream", None),
        ("scan.pdf.exe", "application/pdf", None),
    ],
)
def test_attachment_classification(filename: str, mime: str, expected: ScanFormat | None) -> None:
    assert classify_attachment(filename, mime) is expected


def test_message_with_only_unsupported_attachments_is_ignored() -> None:
    gmail = FakeGmail()
    gmail.add_message("zip", attachments=[("scans.zip", "application/zip", b"PK\x03\x04")])
    assert _parse(gmail, "zip") is None


def test_pdf_preference() -> None:
    gmail = FakeGmail()
    gmail.add_message(
        "mixed", attachments=[("a.png", "image/png", a4_png_bytes()), ("a.pdf", "application/pdf", a4_pdf_bytes())]
    )
    parsed = _parse(gmail, "mixed")
    assert parsed is not None
    assert [item.filename for item in preferred_attachments(parsed)] == ["a.pdf"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("..\\..\\Windows\\evil.pdf", "evil.pdf"),
        ("Scan 2026:10:16 <1>.pdf", "Scan 2026_10_16 _1.pdf"),
        ("....pdf", "scan.pdf"),
        ("report.PDF", "report.pdf"),
        ("noextension", "noextension.pdf"),
    ],
)
def test_filename_sanitization(raw: str, expected: str) -> None:
    assert sanitize_filename(raw, ScanFormat.PDF) == expected


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AttachmentRejectedError):
        safe_destination(tmp_path / "input", "../outside.pdf")


def _save(tmp_path: Path, data: bytes, filename: str = "scan.pdf", mime: str = "application/pdf", max_bytes: int = 10**8):
    return save_attachment_atomically(
        data, filename=filename, mime_type=mime, destination_dir=tmp_path / "input", max_bytes=max_bytes
    )


def test_atomic_save_preserves_bytes_and_leaves_no_temporary_files(tmp_path: Path) -> None:
    data = a4_pdf_bytes()
    saved = _save(tmp_path, data, filename="../../Canon Scan.pdf")
    assert saved.path == (tmp_path / "input" / "Canon Scan.pdf").resolve()
    assert saved.path.read_bytes() == data
    assert saved.pdf_page_count == 1
    assert sorted(path.name for path in (tmp_path / "input").iterdir()) == ["Canon Scan.pdf"]


@pytest.mark.parametrize(
    ("data", "filename", "mime", "message"),
    [
        (b"", "scan.pdf", "application/pdf", "empty"),
        (b"%PDF-" + b"0" * 100, "scan.pdf", "application/pdf", "above"),
        (b"PK\x03\x04data", "scan.pdf", "application/pdf", "archive"),
        (b"MZ\x90\x00", "scan.pdf", "application/octet-stream", "archive or executable"),
        (b"\x89PNG\r\n\x1a\nxx", "scan.pdf", "application/pdf", "does not match"),
        (b"%PDF-1.4", "scan.zip", "application/zip", "unsupported"),
    ],
)
def test_unsafe_attachments_are_rejected(tmp_path: Path, data: bytes, filename: str, mime: str, message: str) -> None:
    with pytest.raises(AttachmentRejectedError, match=message):
        _save(tmp_path, data, filename, mime, max_bytes=50)
    assert not (tmp_path / "input").exists() or not any((tmp_path / "input").iterdir())


def test_unreadable_or_empty_pdf_is_removed(tmp_path: Path) -> None:
    with pytest.raises(AttachmentRejectedError):
        _save(tmp_path, b"%PDF-1.7 not really a pdf")
    empty = pdfium.PdfDocument.new()
    empty_path = tmp_path / "empty.pdf"
    empty.save(str(empty_path))
    with pytest.raises(AttachmentRejectedError, match="no pages|cannot be opened"):
        _save(tmp_path, empty_path.read_bytes())
    assert not any((tmp_path / "input").iterdir())
