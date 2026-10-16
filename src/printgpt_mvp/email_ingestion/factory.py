"""Build a connected ``EmailScanService`` from local settings.

The Streamlit app and the hardware MCP server both call
``connect_email_scan_service``; tests replace it with a fake-backed service.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .auth import AuthResult, AuthStatus, OAuthBackend, load_credentials
from .client import GmailApi, GmailApiError, GoogleGmailApi
from .service import EmailScanService
from .settings import EmailIngestionSettings


@dataclass(frozen=True, slots=True)
class EmailConnection:
    auth: AuthResult
    service: EmailScanService | None
    message: str

    @property
    def connected(self) -> bool:
        return self.service is not None


def load_settings(dotenv_path: Path | None = Path(".env")) -> EmailIngestionSettings:
    """Read settings from ``.env`` and the environment (raises ``ValidationError`` if incomplete)."""
    return EmailIngestionSettings.from_environment(dotenv_path=dotenv_path)


def _masked(address: str) -> str:
    name, _, domain = address.partition("@")
    return f"{name[:2]}***@{domain}" if domain else "***"


def connect_email_scan_service(
    settings: EmailIngestionSettings,
    *,
    allow_browser: bool = False,
    backend: OAuthBackend | None = None,
    gmail_factory: Callable[[object], GmailApi] = GoogleGmailApi,
) -> EmailConnection:
    """Authenticate read-only and confirm the token belongs to the configured inbox."""
    auth = load_credentials(
        settings.credentials_path, settings.token_path, allow_browser=allow_browser, backend=backend
    )
    if not auth.connected:
        return EmailConnection(auth, None, auth.message)
    gmail = gmail_factory(auth.credentials)
    try:
        account = gmail.profile_email()
    except GmailApiError as error:
        return EmailConnection(auth, None, str(error))
    if account != settings.gmail_account:
        # Only a masked form of the unexpected account is shown.
        mismatch = AuthResult(AuthStatus.CONSENT_FAILED, None, auth.message)
        return EmailConnection(
            mismatch, None, f"Signed in as {_masked(account)}, not the configured scan inbox; reconnect with that account."
        )
    return EmailConnection(auth, EmailScanService(settings, gmail), auth.message)
