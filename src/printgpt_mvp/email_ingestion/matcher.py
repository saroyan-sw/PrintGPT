"""Pure rules for recognising scanner emails and choosing the attachment.

Selection rule: only the newest matching message is a candidate. Older
messages are never imported automatically, even if unread; a newest message
that was already processed means "keep waiting for a new scan". Read/unread
state is ignored entirely.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from email.utils import getaddresses, parseaddr
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

from .models import ScanAttachment, ScanFormat, ScanMessage


# Extension and MIME type must agree. Canon devices may label scans as
# octet-stream; those are accepted only when the extension is supported and
# the downloader's content check confirms the format.
_EXTENSIONS: Mapping[str, ScanFormat] = {
    ".pdf": ScanFormat.PDF,
    ".png": ScanFormat.PNG,
    ".jpg": ScanFormat.JPEG,
    ".jpeg": ScanFormat.JPEG,
    ".tif": ScanFormat.TIFF,
    ".tiff": ScanFormat.TIFF,
}
_MIME_TYPES: Mapping[str, ScanFormat] = {
    "application/pdf": ScanFormat.PDF,
    "image/png": ScanFormat.PNG,
    "image/jpeg": ScanFormat.JPEG,
    "image/jpg": ScanFormat.JPEG,
    "image/pjpeg": ScanFormat.JPEG,
    "image/tiff": ScanFormat.TIFF,
}
GENERIC_MIME_TYPES = frozenset({"application/octet-stream", "binary/octet-stream"})


def build_search_query(sender: str, recipient: str, lookback_days: int, subject_filter: str | None) -> str:
    """Narrow Gmail search; exact header checks below still decide matches."""
    parts = [f"from:{sender}", f"to:{recipient}", "has:attachment", f"newer_than:{lookback_days}d"]
    if subject_filter:
        escaped = subject_filter.replace('"', "")
        parts.append(f'subject:"{escaped}"')
    return " ".join(parts)


def classify_attachment(filename: str, mime_type: str) -> ScanFormat | None:
    """Return the scan format when extension and MIME type agree on a supported one."""
    suffix = PurePosixPath(PureWindowsPath(filename).name).suffix.lower()
    by_extension = _EXTENSIONS.get(suffix)
    if by_extension is None:
        return None
    mime = mime_type.split(";", 1)[0].strip().lower()
    if mime in GENERIC_MIME_TYPES:
        return by_extension
    return by_extension if _MIME_TYPES.get(mime) is by_extension else None


def _headers(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for header in payload.get("headers", []) or []:
        name = str(header.get("name", "")).lower()
        result.setdefault(name, []).append(str(header.get("value", "")))
    return result


def _walk_parts(part: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    yield part
    for child in part.get("parts", []) or []:
        yield from _walk_parts(child)


def _addresses(values: Sequence[str]) -> set[str]:
    return {address.lower() for _, address in getaddresses(list(values)) if address}


def _received_at(message: Mapping[str, Any]) -> datetime:
    milliseconds = int(message.get("internalDate", 0))
    return datetime.fromtimestamp(milliseconds / 1000.0, tz=UTC)


def parse_scan_message(
    message: Mapping[str, Any],
    *,
    sender: str,
    recipient: str,
    subject_filter: str | None,
) -> ScanMessage | None:
    """Return a ``ScanMessage`` if this Gmail message is a scanner email, else ``None``.

    Gmail's ``from:``/``to:`` search operators are fuzzy, so the exact sender
    address and a recipient header are re-checked here.
    """
    payload = message.get("payload") or {}
    headers = _headers(payload)
    _, from_address = parseaddr(headers.get("from", [""])[0])
    if from_address.lower() != sender.lower():
        return None
    recipients = _addresses(headers.get("to", []) + headers.get("cc", []) + headers.get("delivered-to", []))
    if recipient.lower() not in recipients:
        return None
    subject = headers.get("subject", [""])[0]
    if subject_filter and not subject.lower().startswith(subject_filter.lower()):
        return None

    message_id = str(message["id"])
    attachments = []
    for part in _walk_parts(payload):
        filename = str(part.get("filename") or "")
        body = part.get("body") or {}
        attachment_id = body.get("attachmentId")
        if not filename or not attachment_id:
            continue
        scan_format = classify_attachment(filename, str(part.get("mimeType", "")))
        if scan_format is None:
            continue
        attachments.append(
            ScanAttachment(
                message_id=message_id,
                attachment_id=str(attachment_id),
                part_id=str(part.get("partId") or ""),
                filename=filename,
                mime_type=str(part.get("mimeType", "")),
                size_bytes=int(body.get("size", 0)),
                scan_format=scan_format,
            )
        )
    if not attachments:
        return None
    return ScanMessage(
        message_id=message_id,
        received_at=_received_at(message),
        subject=subject,
        attachments=tuple(attachments),
    )


def newest_first(messages: Iterable[ScanMessage]) -> tuple[ScanMessage, ...]:
    return tuple(sorted(messages, key=lambda message: (message.received_at, message.message_id), reverse=True))


def preferred_attachments(message: ScanMessage) -> tuple[ScanAttachment, ...]:
    """PDF representations win over images; more than one result needs the user."""
    pdfs = tuple(item for item in message.attachments if item.scan_format is ScanFormat.PDF)
    return pdfs or message.attachments
