"""Streamlit hardware status panel and main-flow print step with an injected fake service."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from hw_fakes import make_service
from printgpt_hardware_mcp.windows_printing import JobRecord
from printgpt_mvp.print_readiness import load_paper_profile


def _panel():
    import streamlit as st
    from printgpt_mvp.hardware_panel import render_hardware_panel

    render_hardware_panel(st)


def _print_step():
    import streamlit as st
    from printgpt_mvp.print_step import render_print_step

    render_print_step(st, st.session_state.get("test_export"))


def _app(function, service, exported=None):
    app = AppTest.from_function(function, default_timeout=60)
    app.session_state["hw_service"] = service
    app.session_state["test_export"] = exported
    return app


def _button(app, label):
    return next(button for button in app.button if button.label == label)


@pytest.fixture
def paper_profile(tmp_path, monkeypatch):
    path = tmp_path / "paper_loading.json"
    monkeypatch.setenv("PRINTGPT_PAPER_PROFILE", str(path))
    return path


def test_panel_only_reports_status(config, printer, scanner, clock):
    app = _app(_panel, make_service(config, printer, scanner, clock)).run()
    assert [b.label for b in app.button] == ["Check printer and scanner"]
    app.button[0].click().run()
    assert any("ready" in m.value for m in app.markdown)
    assert len(app.checkbox) == 0 and printer.submitted == []


def test_print_step_is_hidden_until_approval(config, printer, scanner, clock, paper_profile):
    app = _app(_print_step, make_service(config, printer, scanner, clock)).run()
    assert len(app.button) == 0 and len(app.checkbox) == 0


def test_unverified_loading_needs_a_spare_sheet_then_prints_once(config, printer, scanner, clock, export,
                                                                 paper_profile):
    exported = SimpleNamespace(toner_png_path=Path(export["png"]))
    app = _app(_print_step, make_service(config, printer, scanner, clock), exported).run()
    assert any("not been verified" in w.value for w in app.warning)
    _button(app, "Check printer and prepare the print job").click().run()
    assert not app.exception
    assert sum("✅" in m.value for m in app.markdown) == 6
    print_button = _button(app, "Print one page now")
    assert print_button.disabled
    loaded, spare = app.checkbox[0], app.checkbox[1]
    assert "only sheet in that tray" in loaded.label and "spare test sheet" in spare.label
    loaded.check().run()
    assert _button(app, "Print one page now").disabled
    app.checkbox[1].check().run()
    _button(app, "Print one page now").click().run()
    assert not app.exception
    assert len(printer.submitted) == 1
    assert any("Printed" in s.value for s in app.success)
    _button(app, "The new ink landed correctly: mark this loading direction verified").click().run()
    assert load_paper_profile(paper_profile).verified
    assert len(printer.submitted) == 1


def test_queued_jobs_block_preparation(config, printer, scanner, clock, export, paper_profile):
    printer.jobs.append(JobRecord(7, "someone else's document", 0x0008))
    exported = SimpleNamespace(toner_png_path=Path(export["png"]))
    app = _app(_print_step, make_service(config, printer, scanner, clock), exported).run()
    _button(app, "Check printer and prepare the print job").click().run()
    assert any("❌" in m.value and "No other jobs waiting" in m.value for m in app.markdown)
    assert all(b.label != "Print one page now" for b in app.button)
    assert printer.submitted == []


def test_disabled_physical_printing_is_explained(config, printer, scanner, clock, export, paper_profile):
    service = make_service(replace(config, allow_physical_print=False), printer, scanner, clock)
    exported = SimpleNamespace(toner_png_path=Path(export["png"]))
    app = _app(_print_step, service, exported).run()
    _button(app, "Check printer and prepare the print job").click().run()
    assert any("PRINTGPT_HW_ALLOW_PRINT=1" in m.value for m in app.markdown)
    assert printer.submitted == []
