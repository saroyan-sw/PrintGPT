"""Automatic import of Canon "Scan and Send" emails from Gmail (read-only)."""

from .auth import GMAIL_READONLY_SCOPE, AuthResult, AuthStatus, load_credentials, revoke_local_token
from .factory import EmailConnection, connect_email_scan_service, load_settings
from .models import (
    ImportOutcome,
    ImportResult,
    PollOutcome,
    PollResult,
    ScanAttachment,
    ScanMessage,
    ScanSession,
    ScanSessionStatus,
)
from .service import EmailScanService, UnknownSessionError
from .settings import EmailIngestionSettings

__all__ = [
    "GMAIL_READONLY_SCOPE",
    "AuthResult",
    "AuthStatus",
    "EmailConnection",
    "EmailIngestionSettings",
    "EmailScanService",
    "ImportOutcome",
    "ImportResult",
    "PollOutcome",
    "PollResult",
    "ScanAttachment",
    "ScanMessage",
    "ScanSession",
    "ScanSessionStatus",
    "UnknownSessionError",
    "connect_email_scan_service",
    "load_credentials",
    "load_settings",
    "revoke_local_token",
]
