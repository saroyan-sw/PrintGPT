"""The Streamlit hardware section with an injected fake service (no device access)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from hw_fakes import make_service


def _app():
    import streamlit as st
    from printgpt_mvp.hardware_panel import render_hardware_panel

    render_hardware_panel(st, st.session_state.get("test_export"))


def _panel_app(service, exported):
    app = AppTest.from_function(_app, default_timeout=60)
    app.session_state["hw_service"] = service
    app.session_state["test_export"] = exported
    return app


def test_panel_prepares_but_never_prints_without_confirmation(config, printer, scanner, clock, export):
    service = make_service(config, printer, scanner, clock)
    exported = SimpleNamespace(toner_png_path=Path(export["png"]))
    app = _panel_app(service, exported).run()
    labels = [b.label for b in app.button]
    assert labels == ["Check printer and scanner", "Validate and prepare print job (does not print)"]
    app.button[0].click().run()
    assert any("ready" in m.value for m in app.markdown)
    app.button[1].click().run()
    assert not app.exception and app.success
    print_button = next(b for b in app.button if b.label == "Print one page now")
    assert print_button.disabled
    assert printer.submitted == []


def test_panel_without_export_only_offers_status(config, printer, scanner, clock):
    app = _panel_app(make_service(config, printer, scanner, clock), None).run()
    assert [b.label for b in app.button] == ["Check printer and scanner"]
    assert len(app.checkbox) == 0
