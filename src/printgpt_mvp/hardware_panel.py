"""Optional Streamlit section for the local Canon printer/scanner adapter.

Hardware is touched only when the user presses a button here. Status checks and
job preparation are read-only; the one physical action is separate, clearly
labelled, and requires an explicit confirmation tick.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _service(st: Any):
    service = st.session_state.get("hw_service")
    if service is None:
        from printgpt_hardware_mcp.service import default_service

        service = default_service()
        st.session_state["hw_service"] = service
    return service


def _recent_imports(imports_root: Path, limit: int = 5) -> list[Path]:
    if not imports_root.is_dir():
        return []
    files = [p for p in imports_root.glob("*/input/*") if p.is_file()]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def render_hardware_panel(st: Any, exported: Any | None) -> None:
    with st.expander("Printer and scanner — local hardware adapter (optional)"):
        st.caption(
            "Uses the installed Windows Canon queue and ScanGear driver through the same checks as the "
            "PrintGPT hardware MCP server. Nothing prints automatically; the manual PDF download always works."
        )
        if st.button("Check printer and scanner"):
            try:
                service = _service(st)
                st.session_state["hw_status"] = (service.printer_status().model_dump(),
                                                 service.scanner_status().model_dump())
            except Exception as error:
                st.session_state["hw_status"] = None
                st.error(f"Hardware adapter unavailable: {error}")
        status = st.session_state.get("hw_status")
        if status:
            printer, scanner = status
            state = "ready" if printer["ready"] else ("found, not ready" if printer["found"] else "not found")
            st.write(f"**Printer queue** `{printer['queue']}` — {state} · states: {', '.join(printer['states'])} · "
                     f"jobs queued: {printer['queued_jobs']} · IP reachable: {printer['network_reachable']}")
            mode = "direct WIA scan (experimental)" if scanner["direct_scan_enabled"] else "manual import"
            st.write(f"**Scanner** {scanner['wia_device_name'] or 'not detected'} — device online: "
                     f"{scanner['device_reports_online']} · acquisition: {mode}")
            for note in scanner["notes"]:
                st.caption(note)
            imports = _recent_imports(Path(_service(st).config.imports_root))
            if imports:
                st.write("Recently imported scans: " + ", ".join(f"`{p.parent.parent.name}/{p.name}`" for p in imports))

        if exported is None:
            st.caption("Approve a candidate above to prepare a print job.")
            return
        st.image(str(exported.toner_png_path), caption="Exact toner-only artifact that would be printed (white = no toner)")
        if st.button("Validate and prepare print job (does not print)"):
            st.session_state.pop("hw_submitted", None)
            try:
                service = _service(st)
                report = service.validate_print_artifact(str(exported.toner_png_path))
                st.session_state["hw_validation"] = report.model_dump()
                st.session_state["hw_prepared"] = (service.prepare_print_job(str(exported.toner_png_path)).model_dump()
                                                   if report.valid else None)
            except Exception as error:
                st.session_state["hw_prepared"] = None
                st.error(str(error))
        validation = st.session_state.get("hw_validation")
        if validation:
            if validation["valid"]:
                st.success("Print artifact validation passed.")
            else:
                st.error("Print artifact rejected: " + "; ".join(validation["errors"]))
        submitted = st.session_state.get("hw_submitted")
        if submitted:
            st.success(f"Submitted Windows job {submitted['windows_job_id']} — {submitted['message']}")
        prepared = st.session_state.get("hw_prepared")
        if not prepared:
            return
        st.info(prepared["summary"])
        for warning in prepared["warnings"]:
            st.warning(warning)

        st.markdown("#### Physical print — uses paper and toner")
        enabled = bool(prepared["physical_print_enabled"] and prepared["printer_ready"])
        if not prepared["physical_print_enabled"]:
            st.caption("Physical printing is disabled. Set PRINTGPT_HW_ALLOW_PRINT=1 before starting the app to enable it.")
        confirm = st.checkbox("I am at the printer, the original sheet is loaded, and I want to print this one page now.",
                              key="hw_confirm_print", disabled=not enabled)
        if st.button("Print one page now", disabled=not (enabled and confirm), type="secondary"):
            try:
                result = _service(st).submit_print_job(prepared["preparation_id"], prepared["artifact_sha256"],
                                                       confirm_physical_print=True, dry_run=False)
                st.session_state["hw_submitted"] = result.model_dump()
                st.session_state["hw_prepared"] = None
            except Exception as error:
                st.error(str(error))
