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


def test_uploaded_scan_uses_real_pipeline_and_failed_upload_clears_old_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = np.full((1000, 802, 3), 255, dtype=np.uint8)
    cv2.putText(page, "Scanned form", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.rectangle(page, (80, 300), (710, 800), (0, 0, 0), 3)
    app_path = Path(__file__).resolve().parents[1] / "src/printgpt_mvp/streamlit_app.py"
    with patch("streamlit.file_uploader", return_value=_upload(page)):
        app = AppTest.from_file(str(app_path), default_timeout=45).run()
        app.button[0].click().run()
        assert not app.exception
        assert not app.error
        run = app.session_state["mvp_run"]
        assert run.alignment["alignment_method"] == "scan_frame"
        assert len(run.candidates) == 2
        assert app.success
        app.session_state["approved_export"] = "old export must disappear"
    with patch("streamlit.file_uploader", return_value=_upload(np.full_like(page, 245))):
        app.run()
        app.button[0].click().run()
        assert not app.exception
        assert any("scan again" in error.value.lower() for error in app.error)
        assert "mvp_run" not in app.session_state
        assert "approved_export" not in app.session_state
        assert not app.success
        assert len(app.checkbox) == 0
