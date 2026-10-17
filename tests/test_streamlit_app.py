"""Exercise the actual app's upload -> generation -> failed replacement path."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def _upload(page):
    ok, encoded = cv2.imencode(".png", page)
    assert ok
    upload = BytesIO(encoded.tobytes())
    upload.name = "scanner-export.png"
    return upload


def test_uploaded_scan_requires_preview_confirmation_and_failed_upload_clears_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    cv2.putText(page, "Scanned form", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 0), 10)
    cv2.rectangle(page, (250, 800), (2200, 2800), (0, 0, 0), 10)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=45).run()
        app.button[0].click().run()
        assert not app.exception
        assert not app.error
        assert "input_normalization" in app.session_state
        assert _button(app, "Generate two candidates").disabled
        assert "mvp_run" not in app.session_state
        app.checkbox[0].check().run()
        assert not _button(app, "Generate two candidates").disabled
        _button(app, "Generate two candidates").click().run()
        assert not app.exception
        assert not app.error
        run = app.session_state["mvp_run"]
        assert run.alignment["input_mode"] == "scanner_a4"
        assert run.alignment["marker_calibration_used"] is False
        assert len(run.candidates) == 2
        assert run.alignment["registration"]["status"] == "preview_only"
        assert _button(app, "Generate two candidates").disabled
        assert app.success
        app.session_state["approved_export"] = "old export must disappear"
    with patch("streamlit.file_uploader", return_value=_upload(np.full((600, 600, 3), 245, dtype=np.uint8))):
        app.run()
        app.button[0].click().run()
        assert not app.exception
        assert any("recognizable full-page a4" in error.value.lower() for error in app.error)
        assert "input_normalization" not in app.session_state
        assert "mvp_run" not in app.session_state
        assert "approved_export" not in app.session_state
        assert not app.success
        assert len(app.checkbox) == 0


def test_confirmed_full_page_scan_exposes_registered_export_and_clears_on_change(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    page=np.full((1000,802,3),255,np.uint8)
    cv2.putText(page,"Complete resized A4 sheet",(30,160),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),3)
    cv2.rectangle(page,(60,280),(740,850),(0,0,0),3)
    app_path=Path(__file__).resolve().parents[1]/"src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader",return_value=_upload(page)):
        app=AppTest.from_file(str(app_path),default_timeout=60).run()
        app.selectbox[1].select("full_page").run()
        app.selectbox[2].select(True).run()
        app.selectbox[3].select(True).run()
        app.button[0].click().run()
        assert not app.exception and not app.error
        assert app.session_state["mvp_run"].alignment["registration"]["export_allowed"]
        app.checkbox[0].check().run()
        assert not _button(app, "Approve and create toner-only files").disabled
        _button(app, "Approve and create toner-only files").click().run()
        assert not app.exception and not app.error
        assert "approved_export" in app.session_state
        app.selectbox[2].select(False).run()
        assert "approved_export" not in app.session_state
        assert "mvp_run" not in app.session_state


def test_live_gemini_toggle_without_key_warns_and_blocks_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (250, 800), (2200, 2800), (0, 0, 0), 10)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=45).run()
        assert app.toggle[0].value is False
        app.toggle[0].set_value(True).run()
        assert any("GEMINI_API_KEY" in warning.value for warning in app.warning)
        app.button[0].click().run()
        assert not app.exception
        app.checkbox[0].check().run()
        assert _button(app, "Generate two candidates").disabled
        assert "mvp_run" not in app.session_state
        app.toggle[0].set_value(False).run()
        assert not _button(app, "Generate two candidates").disabled


def test_complete_scanner_sheet_without_measurement_stays_locked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PRINTGPT_SCANNER_PROFILE", str(tmp_path / "no_scanner_profile.json"))
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    cv2.rectangle(page, (250, 800), (2200, 2800), (0, 0, 0), 10)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.button[0].click().run()
        app.checkbox[0].check().run()
        full_sheet = app.checkbox[1]
        assert "complete A4 sheet" in full_sheet.label
        full_sheet.check().run()
        assert any("locked until the scanner placement is measured" in i.value for i in app.info)
        _button(app, "Generate two candidates").click().run()
        assert not app.exception and not app.error
        run = app.session_state["mvp_run"]
        assert run.alignment["registration"]["status"] == "awaiting_scanner_measurement"
        assert any("locked until the scanner placement is measured" in w.value for w in app.warning)


def test_area_is_shown_suggested_and_checked_before_generation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    # A drawing cut off on the right: horizontal strokes all end at x = 1500 px (127 mm).
    for row in range(900, 2400, 150):
        cv2.line(page, (400, row), (1500, row), (0, 0, 0), 8)
    cv2.line(page, (400, 900), (400, 2250), (0, 0, 0), 8)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.button[0].click().run()
        assert any("of the existing drawing" in w.value for w in app.warning)
        assert any("Suggested area" in c.value for c in app.caption)
        _button(app, "Use suggested area").click().run()
        assert app.session_state["region_x"] == pytest.approx(127, abs=1)
        assert not [w.value for w in app.warning if "area" in w.value or "drawing" in w.value]
        assert all(b.label != "Use suggested area" for b in app.button)


def test_live_gemini_needs_an_override_when_the_area_looks_wrong(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    page = np.full((3508, 2480, 3), 255, dtype=np.uint8)
    for row in range(900, 2400, 150):
        cv2.line(page, (400, row), (1500, row), (0, 0, 0), 8)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.toggle[0].set_value(True).run()
        app.button[0].click().run()
        app.checkbox[0].check().run()
        assert _button(app, "Generate two candidates").disabled
        override = next(c for c in app.checkbox if "Generate anyway" in c.label)
        override.check().run()
        assert not _button(app, "Generate two candidates").disabled
        assert "mvp_run" not in app.session_state
