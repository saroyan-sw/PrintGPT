"""Minimal approval UI for the deterministic offline PrintGPT MVP."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from typing import Any

from printgpt_alignment import DEFAULT_LAYOUT

from printgpt_mvp.pipeline import (
    PRINT_AT_ACTUAL_SIZE,
    MvpRun,
    approve_and_export,
    rectangular_allowed_region,
    run_offline_document,
)
from printgpt_mvp.input_normalization import (
    InputMode,
    InputNormalizationResult,
    OrientationChoice,
    normalize_document_input,
)
from printgpt_mvp.email_ingestion import factory as email_factory
from printgpt_mvp.email_ingestion.models import PollOutcome, ScanSessionStatus

_UPLOAD_SOURCE = "upload"
_EMAIL_SOURCE = "email"
_EMAIL_PROGRESS = {
    ScanSessionStatus.WAITING_FOR_EMAIL: "Waiting for scanner email…",
    ScanSessionStatus.NEEDS_SELECTION: "Email found — choose the attachment to import",
    ScanSessionStatus.EMAIL_FOUND: "Email found",
    ScanSessionStatus.DOWNLOADING: "Downloading attachment…",
    ScanSessionStatus.DOWNLOADED: "Attachment downloaded",
    ScanSessionStatus.VALIDATING: "Validating scan…",
    ScanSessionStatus.IMPORTED: "Scan imported",
    ScanSessionStatus.NORMALIZING: "Normalizing A4…",
    ScanSessionStatus.AWAITING_CONFIRMATION: "Scan ready — confirm the preview below",
    ScanSessionStatus.FAILED: "Scan import failed",
    ScanSessionStatus.TIMED_OUT: "No scanner email arrived before the timeout",
    ScanSessionStatus.CANCELLED: "Waiting cancelled",
}


def _candidate_label(run: MvpRun, attempt_id: str) -> str:
    candidate = next(item for item in run.candidates if item.attempt_id == attempt_id)
    return f"Candidate {candidate.candidate_id} — score {candidate.score:.1f}"


def _clear_results() -> None:
    import streamlit as st

    for key in (
        "input_normalization",
        "upload_path",
        "mvp_run",
        "approved_export",
        "approve_preview",
        "normalization_confirmed",
    ):
        st.session_state.pop(key, None)


def _clear_generation() -> None:
    import streamlit as st

    for key in ("mvp_run", "approved_export", "approve_preview"):
        st.session_state.pop(key, None)


def _upload_source(input_mode: InputMode, orientation: OrientationChoice) -> None:
    import streamlit as st

    upload = st.file_uploader(
        "Upload an image or PDF",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
    )

    request_key = (
        _UPLOAD_SOURCE,
        sha256(upload.getvalue()).hexdigest() if upload is not None else None,
        input_mode.value,
        orientation.value,
    )
    if st.session_state.get("request_key") != request_key:
        _clear_results()
        st.session_state["request_key"] = request_key

    if st.button("Prepare normalized preview", type="primary", disabled=upload is None):
        _clear_results()
        try:
            assert upload is not None
            upload_digest = sha256(upload.getvalue()).hexdigest()
            run_root = Path("artifacts/streamlit") / upload_digest[:12]
            suffix = Path(upload.name).suffix.lower() if upload is not None else ".png"
            input_path = run_root / "input" / f"upload{suffix}"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(upload.getvalue())
            with st.spinner("Loading and normalizing the selected input mode…"):
                st.session_state["input_normalization"] = normalize_document_input(
                    input_path,
                    input_mode=input_mode,
                    orientation=orientation,
                    layout=DEFAULT_LAYOUT,
                )
                st.session_state["upload_path"] = input_path
        except Exception as error:
            st.error(str(error))


def _email_connection(settings: Any) -> Any:
    """Connect without a browser first; the Connect button runs Google consent."""
    import streamlit as st

    connection = st.session_state.get("email_connection")
    if connection is None or not connection.connected:
        connection = email_factory.connect_email_scan_service(settings, allow_browser=False)
        st.session_state["email_connection"] = connection
    return connection


def _poll_email_session(service: Any, session_id: str) -> None:
    """Fragment body: one non-blocking inbox check per run."""
    import streamlit as st

    result = service.poll_scan_session(session_id)
    session = result.session
    st.info(_EMAIL_PROGRESS.get(session.status, session.status.value))
    if result.detail and result.outcome is PollOutcome.WAITING:
        st.caption(result.detail)
    if result.outcome is PollOutcome.NEEDS_SELECTION:
        labels = {
            f"{item.message_id}/{item.attachment_id}": f"{item.filename} ({item.mime_type}, {item.size_bytes} bytes)"
            for item in session.selection_candidates
        }
        choice = st.radio("Several scan attachments arrived — choose one", list(labels), format_func=labels.get)
        if st.button("Import selected attachment"):
            message_id, attachment_id = choice.split("/", 1)
            service.import_selected_scan(session_id, message_id, attachment_id)
            st.rerun()
    elif result.outcome is not PollOutcome.WAITING:
        # Leave the fragment so the whole page renders the final state.
        st.rerun()


def _email_scan_source(input_mode: InputMode, orientation: OrientationChoice) -> None:
    import streamlit as st
    from pydantic import ValidationError

    try:
        settings = email_factory.load_settings()
    except ValidationError:
        st.warning(
            "Scanner email is not configured. Set PRINTGPT_GMAIL_ACCOUNT, PRINTGPT_SCAN_SENDER and "
            "PRINTGPT_GMAIL_CREDENTIALS_PATH in the local .env file (see docs/GMAIL_SCAN_INGESTION.md)."
        )
        return

    connection = _email_connection(settings)
    if not connection.connected:
        st.warning(connection.message)
        if st.button("Connect Gmail (read-only)"):
            with st.spinner("Complete Google consent in the browser window that opened…"):
                st.session_state["email_connection"] = email_factory.connect_email_scan_service(
                    settings, allow_browser=True
                )
            st.rerun()
        return
    service = connection.service
    st.success("Gmail connected (read-only).")

    session_id = st.session_state.get("email_scan_session_id")
    session = service.get_scan_session_status(session_id) if session_id else None
    waiting = session is not None and session.status in (
        ScanSessionStatus.WAITING_FOR_EMAIL,
        ScanSessionStatus.NEEDS_SELECTION,
    )
    if not waiting and st.button("Start waiting for scan", type="primary"):
        _clear_results()
        session = service.start_scan_session()
        st.session_state["email_scan_session_id"] = session.session_id
        waiting = True

    if waiting:
        assert session is not None
        st.markdown(
            "1. Place the page on the Canon glass or document feeder.\n"
            "2. Choose **Scan and Send** and send it to the configured Gmail address.\n"
            f"3. Keep this page open. The inbox is checked every {settings.poll_interval_s:.0f} s "
            f"for up to {settings.timeout_s / 60:.0f} minutes."
        )
        st.fragment(run_every=settings.poll_interval_s)(_poll_email_session)(service, session.session_id)
        if st.button("Cancel waiting"):
            service.cancel_scan_session(session.session_id)
            st.rerun()
        return
    if session is None:
        return

    if session.status in (ScanSessionStatus.FAILED, ScanSessionStatus.TIMED_OUT, ScanSessionStatus.CANCELLED):
        st.error(f"{_EMAIL_PROGRESS[session.status]}. {session.error or ''}".strip())
        return
    st.caption(
        f"Imported **{session.original_filename}** ({session.mime_type}, {session.size_bytes} bytes) "
        f"from the scanner email received {session.received_at:%Y-%m-%d %H:%M} UTC."
    )
    request_key = (_EMAIL_SOURCE, session.session_id, input_mode.value, orientation.value)
    if st.session_state.get("request_key") != request_key:
        _clear_results()
        st.session_state["request_key"] = request_key
    if "input_normalization" not in st.session_state:
        try:
            with st.spinner("Normalizing A4…"):
                _, normalization = service.normalize_imported_scan(
                    session.session_id, input_mode=input_mode, orientation=orientation, layout=DEFAULT_LAYOUT
                )
        except (ValueError, IndexError) as error:
            st.error(str(error))
            return
        st.session_state["input_normalization"] = normalization
        st.session_state["upload_path"] = Path(session.imported_path)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="PrintGPT MVP", layout="wide")
    st.title("PrintGPT — safe A4 completion preview")
    st.caption(
        "Choose either a full-page A4 scanner export or a marked calibration page. Output uses A4. "
        "This offline demonstration uses deterministic generator and critic doubles — not live Gemini. "
        "Nothing is exported until you explicitly approve a validated preview."
    )

    mode_labels = {
        InputMode.SCANNER_A4.value: "Scanner A4",
        InputMode.MARKED_PAGE.value: "Marked page (ArUco)",
    }
    input_mode_value = st.selectbox(
        "Input mode",
        list(mode_labels),
        format_func=mode_labels.get,
        help="Scanner A4 explicitly bypasses markers. Marked page always requires the expected ArUco quartet.",
    )
    input_mode = InputMode(input_mode_value)
    orientation_labels = {
        OrientationChoice.AUTO.value: "Auto",
        OrientationChoice.ROTATE_LEFT_90.value: "Rotate left 90°",
        OrientationChoice.ROTATE_RIGHT_90.value: "Rotate right 90°",
        OrientationChoice.ROTATE_180.value: "Rotate 180°",
        OrientationChoice.NO_ROTATION.value: "No rotation",
    }
    if input_mode is InputMode.SCANNER_A4:
        orientation_value = st.selectbox(
            "Orientation",
            list(orientation_labels),
            format_func=orientation_labels.get,
            help="Auto uses the Canon scanner profile for landscape A4. Override it after checking the preview.",
        )
        orientation = OrientationChoice(orientation_value)
    else:
        st.caption("Orientation is determined from the required ArUco marker positions.")
        orientation = OrientationChoice.AUTO

    source = st.radio(
        "Scan source",
        [_UPLOAD_SOURCE, _EMAIL_SOURCE],
        format_func={_UPLOAD_SOURCE: "Upload a file", _EMAIL_SOURCE: "Scanner email (Gmail)"}.get,
        horizontal=True,
        help="Scanner email imports the newest Canon Scan and Send message automatically; upload stays available.",
    )
    if source == _EMAIL_SOURCE:
        _email_scan_source(input_mode, orientation)
    else:
        _upload_source(input_mode, orientation)

    normalization = st.session_state.get("input_normalization")
    if not isinstance(normalization, InputNormalizationResult):
        return

    st.subheader("Corrected canonical A4 preview")
    if normalization.input_mode is InputMode.SCANNER_A4:
        st.info(
            "Scanner-aligned A4. Orientation and dimensions normalized; physical printer registration still requires calibration."
        )
    else:
        st.success(
            "ArUco calibration accepted — "
            f"markers {list(normalization.marker_ids)}, confidence {normalization.confidence:.3f}."
        )
    physical = normalization.source_physical_dimensions_mm
    physical_text = "unknown" if physical is None else f"{physical[0]:.2f} × {physical[1]:.2f} mm"
    st.write(
        f"Source page: **{normalization.source_pixel_dimensions[0]} × "
        f"{normalization.source_pixel_dimensions[1]} px** · physical size: **{physical_text}**"
    )
    st.write(
        f"Detected orientation: **{normalization.detected_orientation}** · applied: "
        f"**{orientation_labels.get(normalization.selected_rotation.value, normalization.selected_rotation.value)}** · "
        f"canonical: **{normalization.canonical_dimensions[0]} × {normalization.canonical_dimensions[1]} px**"
    )
    st.image(normalization.canonical_rgb, caption="Normalized portrait A4 used by masking and generation")
    for warning in normalization.warnings:
        st.warning(warning)
    confirmed_normalization = st.checkbox(
        "I confirm the orientation and placement in this normalized A4 preview look correct.",
        key="normalization_confirmed",
    )

    prompt = st.text_input("Optional continuation prompt", placeholder="Leave blank for automatic continuation")
    st.subheader("Allowed completion rectangle (millimetres from the top-left)")
    columns = st.columns(4)
    x_mm = columns[0].number_input("X", min_value=0.0, max_value=209.0, value=40.0, step=1.0)
    y_mm = columns[1].number_input("Y", min_value=0.0, max_value=296.0, value=60.0, step=1.0)
    width_mm = columns[2].number_input("Width", min_value=1.0, max_value=210.0, value=130.0, step=1.0)
    height_mm = columns[3].number_input("Height", min_value=1.0, max_value=297.0, value=180.0, step=1.0)

    if st.button(
        "Generate two candidates",
        type="primary",
        disabled=not confirmed_normalization,
    ):
        _clear_generation()
        try:
            allowed = rectangular_allowed_region(
                x_mm=x_mm,
                y_mm=y_mm,
                width_mm=width_mm,
                height_mm=height_mm,
            )
            input_path = Path(st.session_state["upload_path"])
            with st.spinner("Protecting existing ink and generating two offline candidates…"):
                st.session_state["mvp_run"] = run_offline_document(
                    input_path,
                    allowed,
                    input_path.parents[1] / "run",
                    user_prompt=prompt,
                    layout=DEFAULT_LAYOUT,
                    input_mode=input_mode,
                    orientation=orientation,
                    normalization_result=normalization,
                    normalization_confirmed=confirmed_normalization,
                )
        except Exception as error:
            st.error(str(error))

    run = st.session_state.get("mvp_run")
    if not isinstance(run, MvpRun):
        return
    if not run.candidates:
        st.error("No candidate passed deterministic validation and criticism. Review the failures below and try again.")
    else:
        st.success(
            f"{len(run.candidates)} candidate(s) are eligible for approval. "
            "The source scan itself is never printable."
        )
        for candidate in run.candidates:
            st.subheader(_candidate_label(run, candidate.attempt_id))
            left, right = st.columns(2)
            if candidate.attempt.normalized is not None:
                left.image(
                    candidate.attempt.normalized.candidate_rgb,
                    caption="Generated complete-page candidate (diagnostic only)",
                )
            if candidate.attempt.preview_rgb is not None:
                right.image(candidate.attempt.preview_rgb, caption="Actual sanitized print preview")
            metrics = candidate.attempt.validation.metrics if candidate.attempt.validation else {}
            st.caption(
                f"Validation accepted · final ink pixels: {metrics.get('final_ink_count', '—')} · "
                f"removed fraction: {float(metrics.get('removed_fraction', 0.0)):.3f}"
            )

        attempt_ids = [candidate.attempt_id for candidate in run.candidates]
        choice = st.radio(
            "Candidate to approve",
            attempt_ids,
            format_func=lambda value: _candidate_label(run, value),
        )
        confirmed = st.checkbox("I approve this sanitized new-ink preview for A4 export.", key="approve_preview")
        if st.button("Approve and create toner-only files", disabled=not confirmed):
            try:
                st.session_state["approved_export"] = approve_and_export(run, choice, approved=True)
            except Exception as error:
                st.error(str(error))

    with st.expander("Validation failures and revisions"):
        failures = False
        for attempt in run.generation.attempts:
            reasons = list(attempt.validation.reasons) if attempt.validation and attempt.validation.reasons else []
            if reasons or attempt.generation.failure or (
                attempt.normalization and attempt.normalization.failure_reasons
            ):
                failures = True
                st.write(f"**{attempt.attempt_id} — {attempt.status.value}**")
                for reason in reasons:
                    st.write(f"- {reason}")
                if attempt.generation.failure:
                    st.write(f"- {attempt.generation.failure.message}")
                if attempt.normalization:
                    for reason in attempt.normalization.failure_reasons:
                        st.write(f"- {reason}")
        if not failures:
            st.write("No validation failures were recorded.")

    exported = st.session_state.get("approved_export")
    if exported is not None:
        st.warning(PRINT_AT_ACTUAL_SIZE)
        st.download_button(
            "Download exact-size A4 PDF",
            data=exported.a4_pdf_path.read_bytes(),
            file_name=exported.a4_pdf_path.name,
            mime="application/pdf",
        )
        st.download_button(
            "Download toner-only PNG",
            data=exported.toner_png_path.read_bytes(),
            file_name=exported.toner_png_path.name,
            mime="image/png",
        )


if __name__ == "__main__":
    main()
