"""Exercise the actual app's upload -> generation -> failed replacement path."""
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


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
        assert app.button[1].disabled
        assert "mvp_run" not in app.session_state
        app.checkbox[0].check().run()
        assert not app.button[1].disabled
        app.button[1].click().run()
        assert not app.exception
        assert not app.error
        run = app.session_state["mvp_run"]
        assert run.alignment["input_mode"] == "scanner_a4"
        assert run.alignment["marker_calibration_used"] is False
        assert len(run.candidates) == 2
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
