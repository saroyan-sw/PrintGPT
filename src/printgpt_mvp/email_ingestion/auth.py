"""Read-only Gmail OAuth for an installed desktop application.

Only ``gmail.readonly`` is ever requested. The first connection opens the
user's browser for Google consent; afterwards the stored refresh token is
reused. Tokens live in the per-user application-data directory, never in the
repository, and are never logged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import json
import os
from pathlib import Path
from typing import Any, Final, Protocol

from .store import write_text_atomically


GMAIL_READONLY_SCOPE: Final[str] = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES: Final[tuple[str, ...]] = (GMAIL_READONLY_SCOPE,)


class AuthStatus(StrEnum):
    CONNECTED = "connected"
    NEEDS_CONSENT = "needs_consent"
    MISSING_CLIENT_SECRETS = "missing_client_secrets"
    INVALID_CLIENT_SECRETS = "invalid_client_secrets"
    CONSENT_FAILED = "consent_failed"


class OAuthCredentials(Protocol):
    """The parts of ``google.oauth2.credentials.Credentials`` used here."""

    @property
    def valid(self) -> bool: ...
    @property
    def expired(self) -> bool: ...
    @property
    def refresh_token(self) -> str | None: ...
    @property
    def scopes(self) -> Any: ...

    def refresh(self, request: Any) -> None: ...
    def to_json(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AuthResult:
    status: AuthStatus
    credentials: OAuthCredentials | None
    message: str

    @property
    def connected(self) -> bool:
        return self.status is AuthStatus.CONNECTED and self.credentials is not None


@dataclass(frozen=True, slots=True)
class OAuthBackend:
    """Google library entry points, injectable so tests never touch Google."""

    credentials_from_file: Callable[[str, list[str]], OAuthCredentials]
    run_consent_flow: Callable[[str, list[str]], OAuthCredentials]
    new_request: Callable[[], Any]
    refresh_errors: tuple[type[BaseException], ...]


def google_oauth_backend() -> OAuthBackend:
    """Real backend; imports the Google libraries only when a connection is made."""
    from google.auth.exceptions import RefreshError, TransportError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    def run_consent_flow(client_secrets_path: str, scopes: list[str]) -> OAuthCredentials:
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets_path, scopes)
        return flow.run_local_server(port=0, open_browser=True, prompt="consent")

    return OAuthBackend(
        credentials_from_file=Credentials.from_authorized_user_file,
        run_consent_flow=run_consent_flow,
        new_request=Request,
        refresh_errors=(RefreshError, TransportError),
    )


def _has_only_readonly_scope(credentials: OAuthCredentials) -> bool:
    scopes = set(credentials.scopes or ())
    return scopes == {GMAIL_READONLY_SCOPE}


def save_token(credentials: OAuthCredentials, token_path: Path) -> None:
    """Write the token atomically and restrict it to the current user where possible."""
    write_text_atomically(token_path, credentials.to_json())
    try:
        os.chmod(token_path, 0o600)
    except OSError:
        # Windows ignores POSIX modes; the per-user AppData location provides isolation.
        pass


def _load_stored(token_path: Path, backend: OAuthBackend) -> OAuthCredentials | None:
    if not token_path.is_file():
        return None
    try:
        credentials = backend.credentials_from_file(str(token_path), list(SCOPES))
    except (ValueError, json.JSONDecodeError):
        return None
    return credentials if _has_only_readonly_scope(credentials) else None


def _client_secrets_problem(path: Path | None) -> AuthResult | None:
    if path is None or not path.is_file():
        return AuthResult(
            AuthStatus.MISSING_CLIENT_SECRETS,
            None,
            "OAuth client file not found; set PRINTGPT_GMAIL_CREDENTIALS_PATH to the downloaded desktop-client JSON.",
        )
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        content = None
    if not isinstance(content, dict) or "installed" not in content:
        return AuthResult(
            AuthStatus.INVALID_CLIENT_SECRETS,
            None,
            "OAuth client file is not a Google 'Desktop app' client JSON.",
        )
    return None


def load_credentials(
    credentials_path: Path | None,
    token_path: Path,
    *,
    allow_browser: bool,
    backend: OAuthBackend | None = None,
) -> AuthResult:
    """Return usable read-only credentials, refreshing or (if allowed) running consent."""
    backend = backend or google_oauth_backend()
    stored = _load_stored(token_path, backend)
    if stored is not None and stored.valid:
        return AuthResult(AuthStatus.CONNECTED, stored, "Gmail connected (read-only).")
    if stored is not None and stored.expired and stored.refresh_token:
        try:
            stored.refresh(backend.new_request())
        except backend.refresh_errors:
            stored = None
        else:
            save_token(stored, token_path)
            return AuthResult(AuthStatus.CONNECTED, stored, "Gmail connected (token refreshed).")

    problem = _client_secrets_problem(credentials_path)
    if problem is not None:
        return problem
    if not allow_browser:
        return AuthResult(AuthStatus.NEEDS_CONSENT, None, "Gmail is not connected yet; connect to give read-only access.")
    assert credentials_path is not None
    try:
        credentials = backend.run_consent_flow(str(credentials_path), list(SCOPES))
    except backend.refresh_errors as error:
        return AuthResult(AuthStatus.CONSENT_FAILED, None, f"Google consent failed: {type(error).__name__}")
    if not _has_only_readonly_scope(credentials):
        return AuthResult(AuthStatus.CONSENT_FAILED, None, "Google granted unexpected scopes; access was not stored.")
    save_token(credentials, token_path)
    return AuthResult(AuthStatus.CONNECTED, credentials, "Gmail connected (read-only).")


def revoke_local_token(token_path: Path) -> bool:
    """Delete the stored token; Google-side revocation is documented separately."""
    if token_path.is_file():
        token_path.unlink()
        return True
    return False
