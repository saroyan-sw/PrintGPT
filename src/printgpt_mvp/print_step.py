"""Main-flow Streamlit step: load the original sheet, check the printer, print one page.

Uses the same ``HardwareService`` checks as the hardware MCP server: artifact
validation, preparation, and a physical-print switch. Nothing prints without
the printer checks passing and an explicit confirmation of how the sheet is
loaded. While the paper-loading profile is unverified, the confirmation asks for
a spare test sheet instead of the original.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from printgpt_mvp.hardware_panel import service_for
from printgpt_mvp.print_readiness import (
    FACES,
    LEADING_EDGES,
    PrintReadiness,
    check_print_readiness,
    load_paper_profile,
    mark_verified,
    save_paper_profile,
    with_loading,
)

_KEYS = ("ps_readiness", "ps_prepared", "ps_submitted", "ps_error", "ps_confirm_loaded", "ps_confirm_spare")


def clear_print_step(st: Any) -> None:
    for key in _KEYS:
        st.session_state.pop(key, None)


def _render_profile(st: Any):
    profile = st.session_state.get("ps_profile")
    if profile is None:
        try:
            profile = load_paper_profile()
        except (OSError, ValueError, TypeError) as error:
            st.error(f"Paper-loading profile could not be read ({error}); using the unverified default.")
            from printgpt_mvp.print_readiness import PaperLoadingProfile

            profile = PaperLoadingProfile()
        st.session_state["ps_profile"] = profile

    st.markdown("**1. Load your original sheet**")
    for number, step in enumerate(profile.instructions(), start=1):
        st.write(f"{number}. {step}")
    if profile.verified:
        st.caption(f"Loading direction verified on {profile.verified_at}.")
    else:
        st.warning(
            "This loading direction has not been verified on this printer yet. Use a spare test sheet with a "
            "copy of the drawing first; after the print, check that the new ink landed on the right side and "
            "the right way up, then mark the direction as verified."
        )
    with st.expander("Change how the sheet is loaded"):
        tray = st.text_input("Tray", value=profile.tray, key="ps_tray")
        face = st.selectbox("Drawing side", FACES, index=FACES.index(profile.face),
                            format_func=lambda value: f"facing {value}", key="ps_face")
        edge = st.selectbox("Edge fed first", LEADING_EDGES, index=LEADING_EDGES.index(profile.leading_edge),
                            format_func=lambda value: f"{value} edge of the drawing", key="ps_edge")
        if st.button("Save loading direction", key="ps_save_profile"):
            updated = with_loading(profile, tray=tray, face=face, leading_edge=edge)
            save_paper_profile(updated)
            st.session_state["ps_profile"] = updated
            clear_print_step(st)
            st.rerun()
    return profile


def _render_checks(st: Any, readiness: PrintReadiness) -> None:
    for check in readiness.checks:
        mark = "✅" if check.passed else "❌"
        st.write(f"{mark} **{check.name}** — {check.detail}")


def render_print_step(st: Any, exported: Any | None) -> None:
    """Shown after approval; the only place in the main flow that can print."""
    if exported is None:
        return
    st.subheader("Print onto your original")
    st.image(str(exported.toner_png_path), caption="Exactly what will be printed (white = no toner)")
    profile = _render_profile(st)

    st.markdown("**2. Check the printer**")
    if st.button("Check printer and prepare the print job", key="ps_check"):
        clear_print_step(st)
        try:
            service = service_for(st)
            status = service.printer_status().model_dump()
            readiness = check_print_readiness(status, physical_print_enabled=service.config.allow_physical_print)
            st.session_state["ps_readiness"] = readiness
            if readiness.ready:
                report = service.validate_print_artifact(str(exported.toner_png_path))
                if report.valid:
                    st.session_state["ps_prepared"] = service.prepare_print_job(str(exported.toner_png_path)).model_dump()
                else:
                    st.session_state["ps_error"] = "Print artifact rejected: " + "; ".join(report.errors)
        except Exception as error:
            st.session_state["ps_error"] = f"Printer check failed: {error}"
    readiness = st.session_state.get("ps_readiness")
    if readiness is not None:
        _render_checks(st, readiness)
        if not readiness.ready:
            st.error("Fix the items marked ❌, then check again.")
    if st.session_state.get("ps_error"):
        st.error(st.session_state["ps_error"])

    submitted = st.session_state.get("ps_submitted")
    if submitted is not None:
        st.success(f"Printed: Windows job {submitted['windows_job_id']}. {submitted['message']}")
        st.markdown(
            "**4. Check the result.** Take the sheet out and check that the new drawing sits on the drawn side, "
            "upright, and joins the original. Scanning it again lets you compare placement."
        )
        if not profile.verified and st.button("The new ink landed correctly: mark this loading direction verified",
                                              key="ps_verify"):
            updated = mark_verified(profile, datetime.now(UTC))
            save_paper_profile(updated)
            st.session_state["ps_profile"] = updated
            st.rerun()
        return

    prepared = st.session_state.get("ps_prepared")
    if prepared is None:
        return
    st.info(prepared["summary"])
    for warning in prepared["warnings"]:
        st.warning(warning)

    st.markdown("**3. Confirm and print** — uses paper and toner")
    loaded = st.checkbox(profile.confirmation(), key="ps_confirm_loaded")
    spare_ok = True
    if not profile.verified:
        spare_ok = st.checkbox(
            "This is a spare test sheet, not my only original (the loading direction is not verified yet).",
            key="ps_confirm_spare",
        )
    can_print = bool(prepared["physical_print_enabled"] and prepared["printer_ready"] and loaded and spare_ok)
    if st.button("Print one page now", type="primary", disabled=not can_print, key="ps_print"):
        try:
            result = service_for(st).submit_print_job(
                prepared["preparation_id"], prepared["artifact_sha256"], confirm_physical_print=True, dry_run=False
            )
            st.session_state["ps_submitted"] = result.model_dump()
            st.session_state.pop("ps_prepared", None)
            st.rerun()
        except Exception as error:
            st.session_state["ps_error"] = f"Printing failed: {error}"
            st.rerun()
