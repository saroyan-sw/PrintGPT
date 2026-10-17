"""Sidebar panel: print-scan loop calibration rounds and the saved per-printer profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from printgpt_mvp.loop_calibration import CalibrationSession, load_profile, save_profile

_SESSIONS_ROOT = Path("artifacts/calibration")
_EMAIL_SCANS = Path("artifacts/email_scans")


def printer_and_tray() -> tuple[str, str]:
    """The allowlisted printer queue and the tray from the paper-loading profile."""
    from printgpt_mvp.hardware_panel import _hardware_config
    from printgpt_mvp.print_readiness import PaperLoadingProfile, load_paper_profile

    try:
        tray = load_paper_profile().tray
    except (OSError, ValueError, TypeError):
        tray = PaperLoadingProfile().tray
    return _hardware_config().printer_queue, tray


def current_loop_profile() -> dict[str, Any] | None:
    queue, tray = printer_and_tray()
    return load_profile(queue, tray)


def _latest_email_scan() -> Path | None:
    scans = [path for path in _EMAIL_SCANS.glob("*/input/*") if path.is_file()]
    return max(scans, key=lambda path: path.stat().st_mtime) if scans else None


def _measure(st: Any, session: CalibrationSession, scan_path: Path) -> None:
    from printgpt_mvp.input_normalization import InputMode, normalize_document_input

    try:
        page = normalize_document_input(scan_path, input_mode=InputMode.SCANNER_A4).canonical_rgb
        session.add_scan(page, scan_path.name)
        st.session_state.pop("cal_error", None)
    except (ValueError, IndexError, OSError) as error:
        st.session_state["cal_error"] = str(error)


def _rounds_table(st: Any, session: CalibrationSession) -> None:
    rows = []
    for result in session.rounds:
        offset = result["offset_mm_at_centre"]
        rows.append({
            "round": result["round"],
            "printed with": "correction" if result["corrected"] else "no correction",
            "mean error mm": round(result["printed_error"]["mean_mm"], 2),
            "max error mm": round(result["printed_error"]["max_mm"], 2),
            "offset x/y mm": f"{offset[0]:+.2f} / {offset[1]:+.2f}",
            "scale": round(sum(result["axis_scale"]) / 2, 4),
            "problem": "; ".join(result["problems"]) or "",
        })
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def render_calibration_panel(st: Any) -> None:
    with st.sidebar:
        st.header("Printer calibration")
        queue, tray = printer_and_tray()
        profile = load_profile(queue, tray)
        if profile is not None:
            st.success(
                f"Calibrated {profile['measured_at'][:16].replace('T', ' ')} · verified error "
                f"{profile['accuracy_bound_mm']:.2f} mm · {profile['rounds']} round(s)"
            )
        else:
            st.info(f"Not calibrated yet for {queue} / {tray}.")
        st.caption(
            "Measures the whole scan → print loop with a marker page: no ruler needed. Each round prints the page "
            "with the latest correction, so the error should shrink and then settle."
        )
        if st.button("Start a new calibration", key="cal_start"):
            st.session_state["cal_session"] = str(CalibrationSession.start(_SESSIONS_ROOT, queue, tray).directory)
            st.session_state.pop("cal_error", None)
        directory = st.session_state.get("cal_session")
        if not directory:
            return
        session = CalibrationSession.load(Path(directory))
        round_number = session.next_round
        st.markdown(f"**Round {round_number}**")
        pdf = session.directory / f"round-{round_number:02d}" / "approved_a4_actual_size.pdf"
        if not pdf.is_file():
            pdf = session.target_for_next_round()
        st.download_button(f"Download round {round_number} page (PDF)", data=pdf.read_bytes(),
                           file_name=f"printgpt_calibration_round_{round_number:02d}.pdf",
                           mime="application/pdf", key=f"cal_pdf_{round_number}")
        st.caption(
            f"1. Print it at **100% / Actual size** (no Fit to page) from the {tray}.  \n"
            "2. Scan the printed sheet with Scan and Send, placed on the glass as for drawings.  \n"
            "3. Measure the scan below."
        )
        latest = _latest_email_scan()
        if latest is not None and st.button(f"Measure latest emailed scan ({latest.name})", key="cal_latest"):
            _measure(st, session, latest)
            st.rerun()
        upload = st.file_uploader("…or upload the scan", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
                                  key="cal_upload")
        if upload is not None and st.button("Measure uploaded scan", key="cal_measure_upload"):
            target = session.directory / f"round-{round_number:02d}" / f"scan{Path(upload.name).suffix.lower()}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(upload.getvalue())
            _measure(st, session, target)
            st.rerun()
        if st.session_state.get("cal_error"):
            st.error(st.session_state["cal_error"])
        _rounds_table(st, session)
        verified = session.verified_round()
        if verified is not None:
            if st.button(f"Save calibration for {queue}", type="primary", key="cal_save"):
                save_profile(session.profile())
                st.rerun()
        elif session.rounds:
            st.caption("Print and measure at least one corrected round within 0.5 mm to save the calibration.")
