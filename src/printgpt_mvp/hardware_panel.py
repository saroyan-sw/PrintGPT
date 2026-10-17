"""Optional Streamlit section showing the local Canon printer/scanner status.

Hardware is touched only when the user presses a button here, and only to read
status. Printing happens in the main flow (``print_step``), after approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _hardware_config():
    """Hardware settings from the environment, falling back to the local ``.env``."""
    import os

    from printgpt_hardware_mcp.config import HardwareConfig
    from printgpt_mvp.email_ingestion.settings import read_dotenv

    return HardwareConfig.from_env({**read_dotenv(Path(".env")), **os.environ})


def service_for(st: Any):
    """The session's ``HardwareService``; tests inject a fake as ``hw_service``.

    A service the app created itself is rebuilt when the settings change (for
    example ``PRINTGPT_HW_ALLOW_PRINT`` in ``.env``), so no restart is needed.
    """
    service = st.session_state.get("hw_service")
    if service is not None and not st.session_state.get("hw_service_owned"):
        return service
    config = _hardware_config()
    if service is None or service.config != config:
        from printgpt_hardware_mcp.service import default_service

        service = default_service(config)
        st.session_state["hw_service"] = service
        st.session_state["hw_service_owned"] = True
    return service


def _recent_imports(imports_root: Path, limit: int = 5) -> list[Path]:
    if not imports_root.is_dir():
        return []
    files = [p for p in imports_root.glob("*/input/*") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def render_hardware_panel(st: Any) -> None:
    with st.expander("Printer and scanner status — local hardware adapter (optional)"):
        st.caption(
            "Reads the installed Windows Canon queue and ScanGear driver through the same checks as the "
            "PrintGPT hardware MCP server. Nothing is printed from here."
        )
        if st.button("Check printer and scanner"):
            try:
                service = service_for(st)
                st.session_state["hw_status"] = (service.printer_status().model_dump(),
                                                 service.scanner_status().model_dump())
            except Exception as error:
                st.session_state["hw_status"] = None
                st.error(f"Hardware adapter unavailable: {error}")
        status = st.session_state.get("hw_status")
        if not status:
            return
        printer, scanner = status
        state = "ready" if printer["ready"] else ("found, not ready" if printer["found"] else "not found")
        st.write(f"**Printer queue** `{printer['queue']}` — {state} · states: {', '.join(printer['states'])} · "
                 f"jobs queued: {printer['queued_jobs']} · IP reachable: {printer['network_reachable']}")
        mode = "direct WIA scan (experimental)" if scanner["direct_scan_enabled"] else "manual import"
        st.write(f"**Scanner** {scanner['wia_device_name'] or 'not detected'} — device online: "
                 f"{scanner['device_reports_online']} · acquisition: {mode}")
        for note in scanner["notes"]:
            st.caption(note)
        imports = _recent_imports(Path(service_for(st).config.imports_root))
        if imports:
            st.write("Recently imported scans: " + ", ".join(f"`{p.parent.parent.name}/{p.name}`" for p in imports))
