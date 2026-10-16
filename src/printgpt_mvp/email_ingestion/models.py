"""Typed scan-session, candidate, and result models.

Sessions are Pydantic models because they are persisted to disk and reloaded
after Streamlit reruns or by another process (the hardware MCP server).
Updates return new copies; nothing is mutated in place.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ScanSessionStatus(StrEnum):
    WAITING_FOR_EMAIL = "waiting_for_email"
    NEEDS_SELECTION = "needs_selection"
    EMAIL_FOUND = "email_found"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VALIDATING = "validating"
    IMPORTED = "imported"
    NORMALIZING = "normalizing"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {
        ScanSessionStatus.IMPORTED,
        ScanSessionStatus.AWAITING_CONFIRMATION,
        ScanSessionStatus.FAILED,
        ScanSessionStatus.TIMED_OUT,
        ScanSessionStatus.CANCELLED,
    }
)
POLLABLE_STATUSES = frozenset({ScanSessionStatus.WAITING_FOR_EMAIL, ScanSessionStatus.NEEDS_SELECTION})


class ScanFormat(StrEnum):
    PDF = "pdf"
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScanAttachment(_Frozen):
    """One supported attachment of a matching message, before download."""

    message_id: str
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int
    scan_format: ScanFormat


class ScanMessage(_Frozen):
    """A message from the scanner sender to the recipient with supported attachments."""

    message_id: str
    received_at: datetime
    subject: str
    attachments: tuple[ScanAttachment, ...]
    already_processed: bool = False


class ScanSession(_Frozen):
    """Persistent state of one "wait for the scanner email" session."""

    session_id: str
    started_at: datetime
    expected_recipient: str
    allowed_sender: str
    status: ScanSessionStatus
    timeout_s: float
    message_id: str | None = None
    attachment_id: str | None = None
    received_at: datetime | None = None
    subject: str | None = None
    original_filename: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    imported_path: Path | None = None
    run_id: str | None = None
    selection_candidates: tuple[ScanAttachment, ...] = ()
    timestamps: dict[str, datetime] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def advance(self, status: ScanSessionStatus, at: datetime, **changes: object) -> ScanSession:
        """Return a copy in ``status`` with the transition time recorded."""
        timestamps = dict(self.timestamps)
        timestamps[status.value] = at
        return self.model_copy(update={"status": status, "timestamps": timestamps, **changes})


class PollOutcome(StrEnum):
    WAITING = "waiting"
    IMPORTED = "imported"
    NEEDS_SELECTION = "needs_selection"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    FAILED = "failed"
    NOT_WAITING = "not_waiting"


class ImportOutcome(StrEnum):
    IMPORTED = "imported"
    ALREADY_PROCESSED = "already_processed"
    REJECTED = "rejected"
    NOT_FOUND = "not_found"
    SESSION_CLOSED = "session_closed"


class PollResult(_Frozen):
    session: ScanSession
    outcome: PollOutcome
    detail: str = ""


class ImportResult(_Frozen):
    session: ScanSession
    outcome: ImportOutcome
    detail: str = ""


class ProcessedOutcome(StrEnum):
    IMPORTED = "imported"
    DUPLICATE_CONTENT = "duplicate_content"
    REJECTED = "rejected"


class ProcessedRecord(_Frozen):
    """One handled message; the registry keeps these so nothing is processed twice.

    Rejected messages are recorded too, so a bad scan does not block every
    later session; ``sha256`` and ``run_id`` are empty for those.
    """

    message_id: str
    attachment_id: str
    sha256: str
    run_id: str
    session_id: str
    imported_at: datetime
    outcome: ProcessedOutcome = ProcessedOutcome.IMPORTED
