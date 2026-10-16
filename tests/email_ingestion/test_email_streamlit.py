"""Drive the real Streamlit app's scanner-email path with a fake Gmail inbox."""

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from printgpt_mvp.email_ingestion.auth import AuthResult, AuthStatus
from printgpt_mvp.email_ingestion.factory import EmailConnection

from email_fakes import FakeGmail, make_service


APP = Path(__file__).resolve().parents[2] / "src/printgpt_mvp/streamlit_app.py"


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _labels(app: AppTest) -> list[str]:
    return [button.label for button in app.button]


@pytest.fixture
def email_app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service, gmail, _ = make_service(tmp_path)
    connection = EmailConnection(AuthResult(AuthStatus.CONNECTED, object(), "ok"), service, "ok")
    with (
        patch("printgpt_mvp.email_ingestion.factory.load_settings", return_value=service.settings),
        patch("printgpt_mvp.email_ingestion.factory.connect_email_scan_service", return_value=connection),
    ):
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        app.radio[0].set_value("email").run()
        yield app, gmail, service


def test_email_scan_reaches_confirmed_preview_before_generation(email_app) -> None:
    app, gmail, service = email_app
    assert not app.exception
    assert "Start waiting for scan" in _labels(app)
    assert "Prepare normalized preview" not in _labels(app)  # upload path hidden, still selectable

    _button(app, "Start waiting for scan").click().run()
    session_id = app.session_state["email_scan_session_id"]
    assert any("Waiting for scanner email" in info.value for info in app.info)
    assert "Cancel waiting" in _labels(app)

    # Reruns keep the same persisted session while nothing has arrived.
    app.run()
    assert app.session_state["email_scan_session_id"] == session_id
    assert "input_normalization" not in app.session_state

    gmail.add_scan("canon-scan", minutes=1)
    app.run()
    app.run()
    assert not app.exception
    normalization = app.session_state["input_normalization"]
    assert normalization.canonical_rgb.shape == (3508, 2480, 3)
    assert normalization.input_mode.value == "scanner_a4"
    assert any("Corrected canonical A4 preview" in header.value for header in app.subheader)
    assert service.get_scan_session_status(session_id).status.value == "awaiting_confirmation"

    generate = _button(app, "Generate two candidates")
    assert generate.disabled
    app.checkbox[0].check().run()
    assert not _button(app, "Generate two candidates").disabled
    assert "mvp_run" not in app.session_state


def test_waiting_can_be_cancelled(email_app) -> None:
    app, _, service = email_app
    _button(app, "Start waiting for scan").click().run()
    session_id = app.session_state["email_scan_session_id"]
    _button(app, "Cancel waiting").click().run()
    assert service.get_scan_session_status(session_id).status.value == "cancelled"
    assert any("cancelled" in error.value.lower() for error in app.error)
    assert "Start waiting for scan" in _labels(app)


def test_unconfigured_email_shows_setup_help(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("PRINTGPT_GMAIL_ACCOUNT", "PRINTGPT_SCAN_SENDER"):
        monkeypatch.delenv(name, raising=False)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.radio[0].set_value("email").run()
    assert not app.exception
    assert any("not configured" in warning.value for warning in app.warning)


def test_unused_fake_inbox_is_empty() -> None:
    assert FakeGmail().messages == {}
