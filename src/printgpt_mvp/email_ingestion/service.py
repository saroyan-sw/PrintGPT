"""Public scan-by-email service shared by the Streamlit app and the MCP server.

A session waits for the newest scanner email. Each poll looks only at the
newest message from the configured sender to the configured recipient:

- no matching message, or the newest one was already processed -> keep waiting;
- the newest message has exactly one preferred attachment (PDF first) -> import it;
- it has several equally preferred attachments -> the user must choose.

There is deliberately no "received after the session started" filter, so the
first session after setup may import a scan that was sent earlier; the user
still confirms the normalized preview before anything continues. The same
message or file content is never imported twice.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from printgpt_alignment import A4CalibrationLayout, DEFAULT_LAYOUT
from printgpt_mvp.input_normalization import (
    InputMode,
    InputNormalizationResult,
    OrientationChoice,
    normalize_document_input,
)

from .client import GmailApi, GmailApiError
from .downloader import AttachmentRejectedError, save_attachment_atomically
from .matcher import build_search_query, newest_first, parse_scan_message, preferred_attachments
from .models import (
    POLLABLE_STATUSES,
    ImportOutcome,
    ImportResult,
    PollOutcome,
    PollResult,
    ProcessedOutcome,
    ProcessedRecord,
    ScanAttachment,
    ScanMessage,
    ScanSession,
    ScanSessionStatus,
)
from .settings import EmailIngestionSettings
from .store import IngestionStore, utc_now


_LOG = logging.getLogger("printgpt.email_ingestion")


class UnknownSessionError(ValueError):
    """No session with this ID exists in the state directory."""


class EmailScanService:
    def __init__(
        self,
        settings: EmailIngestionSettings,
        gmail: GmailApi,
        *,
        store: IngestionStore | None = None,
        clock: Callable[[], datetime] = utc_now,
        new_session_id: Callable[[], str] = lambda: f"scan-{uuid4().hex[:12]}",
    ) -> None:
        self._settings = settings
        self._gmail = gmail
        self._store = store or IngestionStore(settings.state_dir)
        self._clock = clock
        self._new_session_id = new_session_id

    @property
    def settings(self) -> EmailIngestionSettings:
        return self._settings

    # -- sessions -----------------------------------------------------------------

    def start_scan_session(self) -> ScanSession:
        now = self._clock()
        session = ScanSession(
            session_id=self._new_session_id(),
            started_at=now,
            expected_recipient=self._settings.gmail_account,
            allowed_sender=self._settings.scan_sender,
            status=ScanSessionStatus.WAITING_FOR_EMAIL,
            timeout_s=self._settings.timeout_s,
            timestamps={ScanSessionStatus.WAITING_FOR_EMAIL.value: now},
        )
        self._store.save_session(session)
        _LOG.info("scan session %s started", session.session_id)
        return session

    def get_scan_session_status(self, session_id: str) -> ScanSession:
        session = self._store.load_session(session_id)
        if session is None:
            raise UnknownSessionError(f"unknown scan session {session_id!r}")
        return session

    def cancel_scan_session(self, session_id: str) -> ScanSession:
        with self._store.locked():
            session = self.get_scan_session_status(session_id)
            if session.status not in POLLABLE_STATUSES:
                return session
            cancelled = session.advance(ScanSessionStatus.CANCELLED, self._clock())
            self._store.save_session(cancelled)
        _LOG.info("scan session %s cancelled", session_id)
        return cancelled

    def _save(self, session: ScanSession, status: ScanSessionStatus, **changes: object) -> ScanSession:
        updated = session.advance(status, self._clock(), **changes)
        self._store.save_session(updated)
        return updated

    # -- matching -----------------------------------------------------------------

    def list_matching_scan_messages(self) -> tuple[ScanMessage, ...]:
        """Matching scanner messages within the lookback window, newest first."""
        settings = self._settings
        query = build_search_query(
            settings.scan_sender, settings.gmail_account, settings.lookback_days, settings.subject_filter
        )
        processed = self._store.processed_message_ids()
        messages = []
        for message_id in self._gmail.list_message_ids(query, settings.max_messages):
            parsed = parse_scan_message(
                self._gmail.get_message(message_id),
                sender=settings.scan_sender,
                recipient=settings.gmail_account,
                subject_filter=settings.subject_filter,
            )
            if parsed is not None:
                messages.append(parsed.model_copy(update={"already_processed": parsed.message_id in processed}))
        return newest_first(messages)

    def poll_scan_session(self, session_id: str) -> PollResult:
        session = self.get_scan_session_status(session_id)
        if session.status is ScanSessionStatus.CANCELLED:
            return PollResult(session=session, outcome=PollOutcome.CANCELLED)
        if session.status not in POLLABLE_STATUSES:
            return PollResult(session=session, outcome=PollOutcome.NOT_WAITING)
        now = self._clock()
        if (now - session.started_at).total_seconds() > session.timeout_s:
            timed_out = self._save(session, ScanSessionStatus.TIMED_OUT, error="no new scanner email before the timeout")
            return PollResult(session=timed_out, outcome=PollOutcome.TIMED_OUT, detail=timed_out.error or "")
        if session.status is ScanSessionStatus.NEEDS_SELECTION:
            return PollResult(session=session, outcome=PollOutcome.NEEDS_SELECTION)

        try:
            messages = self.list_matching_scan_messages()
        except GmailApiError as error:
            # A transient Gmail failure keeps the session waiting; the next poll retries.
            _LOG.warning("scan session %s: %s", session_id, error)
            return PollResult(session=session, outcome=PollOutcome.WAITING, detail=str(error))
        newest = messages[0] if messages else None
        if newest is None:
            return PollResult(session=session, outcome=PollOutcome.WAITING, detail="no scanner email yet")
        if newest.already_processed:
            return PollResult(
                session=session, outcome=PollOutcome.WAITING, detail="latest scan already imported; waiting for a new one"
            )

        choices = preferred_attachments(newest)
        if len(choices) > 1:
            waiting = self._save(
                session,
                ScanSessionStatus.NEEDS_SELECTION,
                message_id=newest.message_id,
                received_at=newest.received_at,
                subject=newest.subject,
                selection_candidates=choices,
            )
            _LOG.info("scan session %s: message %s has %d candidate attachments", session_id, newest.message_id, len(choices))
            return PollResult(session=waiting, outcome=PollOutcome.NEEDS_SELECTION)

        result = self._import(session, newest, choices[0])
        if result.outcome is ImportOutcome.IMPORTED:
            return PollResult(session=result.session, outcome=PollOutcome.IMPORTED)
        if result.outcome is ImportOutcome.ALREADY_PROCESSED:
            return PollResult(session=result.session, outcome=PollOutcome.WAITING, detail=result.detail)
        return PollResult(session=result.session, outcome=PollOutcome.FAILED, detail=result.detail)

    # -- importing ----------------------------------------------------------------

    def import_selected_scan(self, session_id: str, message_id: str, attachment_id: str) -> ImportResult:
        """Import a specific attachment, e.g. after the user chose among candidates."""
        session = self.get_scan_session_status(session_id)
        if session.status not in POLLABLE_STATUSES:
            return ImportResult(session=session, outcome=ImportOutcome.SESSION_CLOSED, detail=session.status.value)
        message = next(
            (item for item in self.list_matching_scan_messages() if item.message_id == message_id), None
        )
        attachment = None if message is None else next(
            (item for item in message.attachments if item.attachment_id == attachment_id), None
        )
        if message is None or attachment is None:
            return ImportResult(session=session, outcome=ImportOutcome.NOT_FOUND, detail="no such scanner attachment")
        return self._import(session, message, attachment)

    def _import(self, session: ScanSession, message: ScanMessage, attachment: ScanAttachment) -> ImportResult:
        with self._store.locked():
            duplicate = self._store.find_processed(message_id=message.message_id)
            if duplicate is not None:
                return ImportResult(
                    session=session,
                    outcome=ImportOutcome.ALREADY_PROCESSED,
                    detail=f"message already handled ({duplicate.outcome.value}, run {duplicate.run_id or 'none'})",
                )
            found = self._save(
                session,
                ScanSessionStatus.EMAIL_FOUND,
                message_id=message.message_id,
                attachment_id=attachment.attachment_id,
                received_at=message.received_at,
                subject=message.subject,
                original_filename=attachment.filename,
                mime_type=attachment.mime_type,
                selection_candidates=(),
            )
            if attachment.size_bytes > self._settings.max_attachment_bytes:
                return self._reject(
                    found, f"attachment is {attachment.size_bytes} bytes, above the limit", message, attachment
                )
            downloading = self._save(found, ScanSessionStatus.DOWNLOADING)
            try:
                data = self._gmail.get_attachment(message.message_id, attachment.attachment_id)
            except GmailApiError as error:
                # Not recorded as processed: a transport failure may succeed next session.
                return self._reject(downloading, str(error))
            downloaded = self._save(downloading, ScanSessionStatus.DOWNLOADED, size_bytes=len(data))
            validating = self._save(downloaded, ScanSessionStatus.VALIDATING)
            return self._store_validated(validating, data, message, attachment)

    def _record(
        self,
        session: ScanSession,
        message: ScanMessage,
        attachment: ScanAttachment,
        outcome: ProcessedOutcome,
        *,
        sha256: str = "",
        run_id: str = "",
    ) -> None:
        self._store.add_processed(
            ProcessedRecord(
                message_id=message.message_id,
                attachment_id=attachment.attachment_id,
                sha256=sha256,
                run_id=run_id,
                session_id=session.session_id,
                imported_at=self._clock(),
                outcome=outcome,
            )
        )

    def _store_validated(
        self, session: ScanSession, data: bytes, message: ScanMessage, attachment: ScanAttachment
    ) -> ImportResult:
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self._store.find_processed(sha256=digest)
        if duplicate is not None:
            # Same scan forwarded or re-sent: remember this message too, then keep waiting.
            self._record(
                session, message, attachment, ProcessedOutcome.DUPLICATE_CONTENT, sha256=digest, run_id=duplicate.run_id
            )
            detail = f"identical file already imported as run {duplicate.run_id}"
            waiting = self._save(
                session,
                ScanSessionStatus.WAITING_FOR_EMAIL,
                message_id=None,
                attachment_id=None,
                warnings=(*session.warnings, detail),
            )
            return ImportResult(session=waiting, outcome=ImportOutcome.ALREADY_PROCESSED, detail=detail)
        run_id = f"email-{digest[:12]}"
        try:
            saved = save_attachment_atomically(
                data,
                filename=attachment.filename,
                mime_type=attachment.mime_type,
                destination_dir=self._settings.runs_root / run_id / "input",
                max_bytes=self._settings.max_attachment_bytes,
            )
        except AttachmentRejectedError as error:
            return self._reject(session, str(error), message, attachment)
        self._record(session, message, attachment, ProcessedOutcome.IMPORTED, sha256=saved.sha256, run_id=run_id)
        imported = self._save(
            session,
            ScanSessionStatus.IMPORTED,
            sha256=saved.sha256,
            size_bytes=saved.size_bytes,
            imported_path=saved.path,
            run_id=run_id,
        )
        _LOG.info(
            "scan session %s imported message %s attachment %s (%s, %d bytes, sha256 %s)",
            session.session_id, message.message_id, saved.path.name, attachment.mime_type, saved.size_bytes, saved.sha256,
        )
        return ImportResult(session=imported, outcome=ImportOutcome.IMPORTED)

    def _reject(
        self,
        session: ScanSession,
        reason: str,
        message: ScanMessage | None = None,
        attachment: ScanAttachment | None = None,
    ) -> ImportResult:
        """Fail the session; a rejected attachment is recorded so it is not retried forever."""
        if message is not None and attachment is not None:
            self._record(session, message, attachment, ProcessedOutcome.REJECTED)
        failed = self._save(session, ScanSessionStatus.FAILED, error=reason)
        _LOG.warning("scan session %s failed: %s", session.session_id, reason)
        return ImportResult(session=failed, outcome=ImportOutcome.REJECTED, detail=reason)

    # -- hand-off to the Scanner A4 pipeline -------------------------------------

    def normalize_imported_scan(
        self,
        session_id: str,
        *,
        input_mode: InputMode = InputMode.SCANNER_A4,
        orientation: OrientationChoice = OrientationChoice.AUTO,
        layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    ) -> tuple[ScanSession, InputNormalizationResult]:
        """Run the existing document normalization on the imported file.

        Scanner A4 is the default; ArUco detection runs only when the caller
        explicitly passes ``InputMode.MARKED_PAGE``. ``ValueError`` from
        normalization (for example a non-A4 scan) marks the session failed and
        is re-raised for the caller to display.
        """
        session = self.get_scan_session_status(session_id)
        if session.status not in (ScanSessionStatus.IMPORTED, ScanSessionStatus.AWAITING_CONFIRMATION):
            raise ValueError(f"session {session_id} has no imported scan (status {session.status.value})")
        assert session.imported_path is not None
        normalizing = self._save(session, ScanSessionStatus.NORMALIZING)
        try:
            result = normalize_document_input(
                Path(session.imported_path), input_mode=input_mode, orientation=orientation, layout=layout
            )
        except (ValueError, IndexError) as error:
            self._save(normalizing, ScanSessionStatus.FAILED, error=f"normalization failed: {error}")
            raise
        ready = self._save(normalizing, ScanSessionStatus.AWAITING_CONFIRMATION, warnings=(*session.warnings, *result.warnings))
        return ready, result
