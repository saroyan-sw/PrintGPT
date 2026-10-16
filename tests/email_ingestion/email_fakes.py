"""Fake Gmail API and fixture scans for email-ingestion tests (no real inbox)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from copy import deepcopy
from functools import lru_cache
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from printgpt_mvp.email_ingestion.client import GmailApiError
from printgpt_mvp.email_ingestion.service import EmailScanService
from printgpt_mvp.email_ingestion.settings import EmailIngestionSettings
from printgpt_mvp.email_ingestion.store import IngestionStore


RECIPIENT = "scan-inbox@example.com"
SENDER = "canon-scanner@example.com"
T0 = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


def a4_page() -> np.ndarray:
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    cv2.circle(page, (1240, 1500), 500, (0, 0, 0), 12)
    cv2.line(page, (700, 2400), (1800, 2400), (0, 0, 0), 12)
    return page


@lru_cache(maxsize=1)
def a4_pdf_bytes() -> bytes:
    # Cached: PIL embeds a timestamp, and duplicate tests need identical bytes.
    buffer = io.BytesIO()
    Image.fromarray(a4_page()).save(buffer, format="PDF", resolution=300.0)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def a4_png_bytes() -> bytes:
    ok, encoded = cv2.imencode(".png", a4_page())
    assert ok
    return encoded.tobytes()


class FakeGmail:
    """In-memory Gmail: search returns every stored message, like a loose query.

    Like real Gmail, every ``get_message`` call issues fresh attachmentIds; only
    the MIME ``partId`` is stable. Downloads are recorded as (message, partId).
    """

    def __init__(self, account: str = RECIPIENT) -> None:
        self.account = account
        self.messages: dict[str, dict[str, Any]] = {}
        self.attachments: dict[tuple[str, str], bytes] = {}
        self.issued_attachment_ids: dict[str, tuple[str, str]] = {}
        self.queries: list[str] = []
        self.attachment_downloads: list[tuple[str, str]] = []
        self.fail_listing = False
        self._fetches = 0

    def add_message(
        self,
        message_id: str,
        *,
        sender: str = SENDER,
        recipient: str = RECIPIENT,
        received_at: datetime = T0,
        subject: str = "Scanned from a Canon",
        attachments: list[tuple[str, str, bytes]] | None = None,
    ) -> None:
        parts: list[dict[str, Any]] = [
            {"mimeType": "text/plain", "filename": "", "body": {"size": 12, "data": "UHJpdmF0ZSBib2R5"}}
        ]
        for index, (filename, mime_type, data) in enumerate(attachments or []):
            part_id = str(index + 1)
            self.attachments[(message_id, part_id)] = data
            parts.append({"partId": part_id, "mimeType": mime_type, "filename": filename, "body": {"size": len(data)}})
        self.messages[message_id] = {
            "id": message_id,
            "internalDate": str(int(received_at.timestamp() * 1000)),
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": f"Canon Scanner <{sender}>"},
                    {"name": "To", "value": recipient},
                    {"name": "Subject", "value": subject},
                ],
                "parts": parts,
            },
        }

    def add_scan(self, message_id: str, *, minutes: int = 0, pdf: bool = True, **kwargs: Any) -> None:
        data = a4_pdf_bytes() if pdf else a4_png_bytes()
        name, mime = ("scan.pdf", "application/pdf") if pdf else ("scan.png", "image/png")
        self.add_message(
            message_id, received_at=T0 + timedelta(minutes=minutes), attachments=[(name, mime, data)], **kwargs
        )

    def profile_email(self) -> str:
        return self.account

    def list_message_ids(self, query: str, max_results: int) -> list[str]:
        self.queries.append(query)
        if self.fail_listing:
            raise GmailApiError("Gmail message search failed with HTTP 503")
        ordered = sorted(self.messages.values(), key=lambda item: int(item["internalDate"]), reverse=True)
        return [item["id"] for item in ordered][:max_results]

    def get_message(self, message_id: str) -> dict[str, Any]:
        self._fetches += 1
        message = deepcopy(self.messages[message_id])
        for part in message["payload"]["parts"]:
            if "partId" in part:
                attachment_id = f"{message_id}-{part['partId']}-fetch{self._fetches}"
                self.issued_attachment_ids[attachment_id] = (message_id, part["partId"])
                part["body"]["attachmentId"] = attachment_id
        return message

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes:
        key = self.issued_attachment_ids[attachment_id]
        assert key[0] == message_id
        self.attachment_downloads.append(key)
        return self.attachments[key]


class SteppingClock:
    def __init__(self, start: datetime = T0) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now = self.now + timedelta(seconds=seconds)


def make_settings(tmp_path: Path, **overrides: Any) -> EmailIngestionSettings:
    values: dict[str, Any] = {
        "gmail_account": RECIPIENT,
        "scan_sender": SENDER,
        "token_path": tmp_path / "appdata" / "token.json",
        "state_dir": tmp_path / "appdata" / "state",
        "runs_root": tmp_path / "artifacts" / "email_scans",
    }
    values.update(overrides)
    return EmailIngestionSettings(**values)


def make_service(
    tmp_path: Path, gmail: FakeGmail | None = None, clock: SteppingClock | None = None, **overrides: Any
) -> tuple[EmailScanService, FakeGmail, SteppingClock]:
    gmail = gmail or FakeGmail()
    clock = clock or SteppingClock()
    settings = make_settings(tmp_path, **overrides)
    counter = iter(range(1, 1000))
    service = EmailScanService(
        settings,
        gmail,
        store=IngestionStore(settings.state_dir),
        clock=clock,
        new_session_id=lambda: f"scan-{next(counter)}",
    )
    return service, gmail, clock
