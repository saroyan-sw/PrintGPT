"""Validated, atomic saving of a downloaded scan attachment.

The bytes are written unchanged: the original attachment is preserved and
normalization works on its own copy in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile

import pypdfium2 as pdfium

from .matcher import classify_attachment
from .models import ScanFormat


class AttachmentRejectedError(Exception):
    """The attachment is not an acceptable scan; the message explains why."""


_MAGIC: dict[ScanFormat, tuple[bytes, ...]] = {
    ScanFormat.PDF: (b"%PDF-",),
    ScanFormat.PNG: (b"\x89PNG\r\n\x1a\n",),
    ScanFormat.JPEG: (b"\xff\xd8\xff",),
    ScanFormat.TIFF: (b"II*\x00", b"MM\x00*"),
}
# Archives and executables are rejected by content even if renamed.
_FORBIDDEN_MAGIC = (b"PK\x03\x04", b"MZ", b"\x7fELF", b"Rar!", b"7z\xbc\xaf", b"\x1f\x8b")
_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")
_MAX_FILENAME_LENGTH = 100


@dataclass(frozen=True, slots=True)
class SavedAttachment:
    path: Path
    sha256: str
    size_bytes: int
    scan_format: ScanFormat
    pdf_page_count: int | None


def sanitize_filename(filename: str, scan_format: ScanFormat) -> str:
    """Return a plain basename with only safe characters and a supported extension."""
    base = PurePosixPath(PureWindowsPath(filename).name).name
    path = PurePosixPath(base)
    suffix = path.suffix.lower().lstrip(".")
    stem = path.stem if suffix else base
    stem = _UNSAFE_CHARACTERS.sub("_", stem).strip(" ._")[: _MAX_FILENAME_LENGTH - 6] or "scan"
    if classify_attachment(f"x.{suffix}", "application/octet-stream") is not scan_format:
        suffix = {"pdf": "pdf", "png": "png", "jpeg": "jpg", "tiff": "tif"}[scan_format.value]
    return f"{stem}.{suffix}"


def safe_destination(directory: Path, filename: str) -> Path:
    """Join and confirm the result stays inside ``directory``."""
    root = directory.resolve()
    destination = (root / filename).resolve()
    if destination.parent != root:
        raise AttachmentRejectedError(f"filename {filename!r} escapes the destination directory")
    return destination


def validate_attachment_bytes(data: bytes, filename: str, mime_type: str, max_bytes: int) -> ScanFormat:
    """Check size, declared type, and actual content before anything is written."""
    if not data:
        raise AttachmentRejectedError("attachment is empty")
    if len(data) > max_bytes:
        raise AttachmentRejectedError(f"attachment is {len(data)} bytes, above the {max_bytes}-byte limit")
    scan_format = classify_attachment(filename, mime_type)
    if scan_format is None:
        raise AttachmentRejectedError(f"unsupported attachment type {mime_type!r} for {filename!r}")
    if data.startswith(_FORBIDDEN_MAGIC):
        raise AttachmentRejectedError("attachment content is an archive or executable")
    if not data.startswith(_MAGIC[scan_format]):
        raise AttachmentRejectedError(f"attachment content does not match its {scan_format.value.upper()} type")
    return scan_format


def _pdf_page_count(data: bytes) -> int:
    # Opened from memory: a path-based document keeps the file locked on Windows,
    # which would stop a rejected file from being removed.
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as error:
        raise AttachmentRejectedError(f"PDF cannot be opened: {error}") from error
    try:
        return len(document)
    finally:
        document.close()


def save_attachment_atomically(
    data: bytes,
    *,
    filename: str,
    mime_type: str,
    destination_dir: Path,
    max_bytes: int,
) -> SavedAttachment:
    """Validate, write to a temporary file, fsync, then rename into place."""
    scan_format = validate_attachment_bytes(data, filename, mime_type, max_bytes)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = safe_destination(destination_dir, sanitize_filename(filename, scan_format))
    digest = hashlib.sha256(data).hexdigest()

    # The temporary file shares the destination directory so the rename is atomic.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".incoming-", suffix=".part", dir=destination_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    written = destination.read_bytes()
    if hashlib.sha256(written).hexdigest() != digest:
        destination.unlink(missing_ok=True)
        raise AttachmentRejectedError("saved file does not match the downloaded bytes")
    page_count = None
    if scan_format is ScanFormat.PDF:
        try:
            page_count = _pdf_page_count(written)
            if page_count < 1:
                raise AttachmentRejectedError("PDF contains no pages")
        except AttachmentRejectedError:
            destination.unlink(missing_ok=True)
            raise
    return SavedAttachment(destination, digest, len(data), scan_format, page_count)
