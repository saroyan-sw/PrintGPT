import json

import pytest
from pydantic import ValidationError

from printgpt_mvp.email_ingestion.auth import (
    GMAIL_READONLY_SCOPE,
    AuthStatus,
    OAuthBackend,
    load_credentials,
    revoke_local_token,
)
from printgpt_mvp.email_ingestion.factory import connect_email_scan_service
from printgpt_mvp.email_ingestion.settings import EmailIngestionSettings, missing_settings

from email_fakes import RECIPIENT, SENDER, FakeGmail, make_settings


class RefreshFailed(Exception):
    pass


class FakeCredentials:
    def __init__(self, *, valid=True, expired=False, refresh_token="refresh", scopes=(GMAIL_READONLY_SCOPE,), fail_refresh=False):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.scopes = list(scopes)
        self.fail_refresh = fail_refresh
        self.refreshed = False

    def refresh(self, request) -> None:
        if self.fail_refresh:
            raise RefreshFailed("expired")
        self.refreshed = True
        self.valid = True
        self.expired = False

    def to_json(self) -> str:
        return json.dumps({"token": "secret-access-token", "refresh_token": self.refresh_token, "scopes": self.scopes})


def _backend(stored=None, consent=None, calls=None):
    calls = calls if calls is not None else []

    def from_file(path, scopes):
        calls.append(("load", scopes))
        if stored is None:
            raise ValueError("no token")
        return stored

    def run_consent(path, scopes):
        calls.append(("consent", scopes))
        return consent or FakeCredentials()

    return OAuthBackend(from_file, run_consent, lambda: object(), (RefreshFailed,)), calls


def _client_secrets(tmp_path, content=None):
    path = tmp_path / "client_secret.json"
    path.write_text(json.dumps(content or {"installed": {"client_id": "x", "client_secret": "y"}}), encoding="utf-8")
    return path


def _token(tmp_path):
    path = tmp_path / "token.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_valid_stored_token_connects_without_browser(tmp_path) -> None:
    backend, calls = _backend(stored=FakeCredentials())
    result = load_credentials(None, _token(tmp_path), allow_browser=False, backend=backend)
    assert result.status is AuthStatus.CONNECTED
    assert calls == [("load", [GMAIL_READONLY_SCOPE])]


def test_expired_token_is_refreshed_and_saved(tmp_path) -> None:
    stored = FakeCredentials(valid=False, expired=True)
    token = _token(tmp_path)
    backend, _ = _backend(stored=stored)
    result = load_credentials(None, token, allow_browser=False, backend=backend)
    assert result.connected and stored.refreshed
    assert json.loads(token.read_text(encoding="utf-8"))["refresh_token"] == "refresh"


def test_failed_refresh_requires_consent(tmp_path) -> None:
    backend, calls = _backend(stored=FakeCredentials(valid=False, expired=True, fail_refresh=True))
    result = load_credentials(_client_secrets(tmp_path), _token(tmp_path), allow_browser=False, backend=backend)
    assert result.status is AuthStatus.NEEDS_CONSENT
    assert ("consent", [GMAIL_READONLY_SCOPE]) not in calls


def test_consent_requests_only_readonly_scope_and_stores_token(tmp_path) -> None:
    token = tmp_path / "appdata" / "token.json"
    backend, calls = _backend()
    result = load_credentials(_client_secrets(tmp_path), token, allow_browser=True, backend=backend)
    assert result.connected
    assert ("consent", [GMAIL_READONLY_SCOPE]) in calls
    assert token.is_file()


def test_unexpected_granted_scope_is_not_stored(tmp_path) -> None:
    token = tmp_path / "token.json"
    backend, _ = _backend(consent=FakeCredentials(scopes=(GMAIL_READONLY_SCOPE, "https://mail.google.com/")))
    result = load_credentials(_client_secrets(tmp_path), token, allow_browser=True, backend=backend)
    assert result.status is AuthStatus.CONSENT_FAILED
    assert not token.exists()


def test_stored_token_with_broader_scope_is_ignored(tmp_path) -> None:
    backend, _ = _backend(stored=FakeCredentials(scopes=("https://mail.google.com/",)))
    result = load_credentials(_client_secrets(tmp_path), _token(tmp_path), allow_browser=False, backend=backend)
    assert result.status is AuthStatus.NEEDS_CONSENT


def test_missing_and_invalid_client_secrets(tmp_path) -> None:
    backend, _ = _backend()
    missing = load_credentials(tmp_path / "nope.json", tmp_path / "token.json", allow_browser=True, backend=backend)
    assert missing.status is AuthStatus.MISSING_CLIENT_SECRETS
    invalid = load_credentials(
        _client_secrets(tmp_path, {"web": {}}), tmp_path / "token.json", allow_browser=True, backend=backend
    )
    assert invalid.status is AuthStatus.INVALID_CLIENT_SECRETS


def test_token_messages_never_contain_secrets(tmp_path) -> None:
    backend, _ = _backend()
    result = load_credentials(_client_secrets(tmp_path), tmp_path / "token.json", allow_browser=True, backend=backend)
    assert "secret-access-token" not in result.message


def test_revoke_local_token(tmp_path) -> None:
    token = _token(tmp_path)
    assert revoke_local_token(token)
    assert not revoke_local_token(token)


def test_connection_rejects_a_different_gmail_account(tmp_path) -> None:
    settings = make_settings(tmp_path, token_path=_token(tmp_path))
    backend, _ = _backend(stored=FakeCredentials())
    connection = connect_email_scan_service(
        settings, backend=backend, gmail_factory=lambda credentials: FakeGmail(account="someone.else@example.com")
    )
    assert not connection.connected
    assert "someone.else" not in connection.message
    good = connect_email_scan_service(settings, backend=backend, gmail_factory=lambda credentials: FakeGmail())
    assert good.connected


def test_settings_from_dotenv_and_environment(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        f"# local only\nPRINTGPT_GMAIL_ACCOUNT={RECIPIENT.upper()}\nPRINTGPT_SCAN_SENDER=\"{SENDER}\"\n"
        "PRINTGPT_SCAN_SUBJECT_FILTER=PrintGPT Scan\n",
        encoding="utf-8",
    )
    settings = EmailIngestionSettings.from_environment({"PRINTGPT_SCAN_TIMEOUT_S": "240"}, dotenv)
    assert settings.gmail_account == RECIPIENT
    assert settings.scan_sender == SENDER
    assert settings.subject_filter == "PrintGPT Scan"
    assert settings.timeout_s == 240
    assert settings.token_path.name == "gmail_token.json"
    assert "PrintGPT" in settings.token_path.parts


def test_missing_settings_are_reported(tmp_path) -> None:
    assert missing_settings({}, None) == ["PRINTGPT_GMAIL_ACCOUNT", "PRINTGPT_SCAN_SENDER"]
    with pytest.raises(ValidationError):
        EmailIngestionSettings.from_environment({"PRINTGPT_GMAIL_ACCOUNT": "not-an-address"}, None)
