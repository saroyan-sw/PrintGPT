"""Minimal approval UI for the PrintGPT MVP (offline doubles by default, live Gemini on request)."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json
import cv2
from typing import Any

from printgpt_alignment import DEFAULT_LAYOUT
from printgpt_alignment.document import load_document
from printgpt_alignment.registration import RegistrationOptions

from printgpt_mvp.hardware_panel import render_hardware_panel
from printgpt_mvp.print_step import render_print_step
from printgpt_mvp.scanner_profile import measured_scanner_profile
from printgpt_mvp.pipeline import (
    PRINT_AT_ACTUAL_SIZE,
    GenerationBackend,
    LiveGenerationUnavailable,
    MvpRun,
    approve_and_export,
    canonicalize_document,
    gemini_api_key,
    rectangular_allowed_region,
    run_offline_document,
)
from printgpt_mvp.input_normalization import (
    AppliedTransform,
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
        "generation_completed_for_preview",
    ):
        st.session_state.pop(key, None)


def _clear_generation() -> None:
    import streamlit as st

    for key in (
        "mvp_run",
        "approved_export",
        "approve_preview",
        "generation_completed_for_preview",
        "ps_readiness",
        "ps_prepared",
        "ps_submitted",
        "ps_error",
        "ps_confirm_loaded",
        "ps_confirm_spare",
    ):
        st.session_state.pop(key, None)


def _registered_preview(
    input_path: Path,
    *,
    page_index: int,
    input_kind: str,
    registration: RegistrationOptions,
) -> InputNormalizationResult:
    """Represent explicit physical registration in the shared preview model."""
    canonical, alignment = canonicalize_document(
        input_path,
        page_index=page_index,
        layout=DEFAULT_LAYOUT,
        input_kind=input_kind,
        registration=registration,
    )
    source = load_document(input_path, page_index=page_index, dpi=DEFAULT_LAYOUT.dpi)
    height, width = source.image.shape[:2]
    matrix = alignment["source_to_canonical"]
    registered = bool(alignment.get("registration", {}).get("export_allowed"))
    marker_ids = tuple(int(value) for value in alignment.get("detected_ids", ()))
    return InputNormalizationResult(
        canonical_rgb=canonical,
        input_mode=InputMode.MARKED_PAGE if marker_ids else InputMode.SCANNER_A4,
        source_pixel_dimensions=(width, height),
        source_physical_dimensions_mm=tuple(alignment["registration"]["page_size_mm"]),
        requested_orientation=OrientationChoice.AUTO,
        selected_rotation=OrientationChoice.NO_ROTATION,
        detected_orientation=f"physical registration: {registration.mode}",
        applied_transform=AppliedTransform(
            operation=f"physical_registration_{registration.mode}",
            matrix=tuple(tuple(float(value) for value in row) for row in matrix),
            source_dimensions=(width, height),
            rotated_dimensions=(width, height),
            resized_dimensions=DEFAULT_LAYOUT.raster_size_px,
            destination_dimensions=DEFAULT_LAYOUT.raster_size_px,
            scale_xy=(
                DEFAULT_LAYOUT.raster_width_px / width,
                DEFAULT_LAYOUT.raster_height_px / height,
            ),
            padding_ltrb=(0, 0, 0, 0),
            interpolation="INTER_LINEAR",
        ),
        canonical_dimensions=DEFAULT_LAYOUT.raster_size_px,
        confidence=1.0 if registered else 0.7,
        confirmation_required=True,
        warnings=tuple(alignment.get("warnings", ())),
        marker_calibration_used=bool(marker_ids),
        marker_ids=marker_ids,
        physical_coordinates_calibrated=registered,
        pdf_rotation_degrees=source.metadata.get("rotation_degrees"),
    )


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
            f"{item.message_id}/{item.part_id}": f"{item.filename} ({item.mime_type}, {item.size_bytes} bytes)"
            for item in session.selection_candidates
        }
        choice = st.radio("Several scan attachments arrived — choose one", list(labels), format_func=labels.get)
        if st.button("Import selected attachment"):
            message_id, part_id = choice.split("/", 1)
            service.import_selected_scan(session_id, message_id, part_id)
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
        "By default the generator and critic are deterministic offline doubles; switch on Live Gemini "
        "below to use the real models. Nothing is exported until you explicitly approve a validated preview."
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
    with st.expander("Physical page registration — required for printing onto the original"):
        st.write("A4-shaped pixels or PDF dimensions alone cannot prove that a scan is uncropped and correctly placed.")
        page_number = st.number_input("PDF page number", min_value=1, value=1, step=1)
        modes = {
            "auto": "Preview / automatic ArUco registration",
            "full_page": "Complete sheet: image edges ARE the paper edges",
            "corners": "Correct paper corners (borders, rotation or perspective)",
            "metadata": "Use PDF page geometry or scanner DPI",
            "crop": "Cropped scan: known DPI and measured paper offset",
        }
        mode = st.selectbox("Registration method", list(modes), format_func=modes.get)
        confirmed_geometry = st.selectbox(
            "Geometry confirmation",
            [False, True],
            format_func=lambda value: (
                "I verified the paper size, orientation, scale and origin"
                if value
                else "Not verified — preview only"
            ),
        )
        paper_width = st.number_input("Actual paper width (mm)", min_value=1.0, value=210.0)
        paper_height = st.number_input("Actual paper height (mm)", min_value=1.0, value=297.0)
        corners_text = st.text_area(
            "Paper corners in source pixels: [[TL x,y],[TR x,y],[BR x,y],[BL x,y]]",
            value="",
            help=(
                "Use the true paper edges, not the text box. Order by the physical sheet's reading "
                "orientation. Inspect source pixels in the geometry panel after a preview run."
            ),
        )
        nonuniform = st.selectbox(
            "Complete sheet was nonuniformly resized",
            [False, True],
            help="Only enable if the whole sheet is present and has been stretched. This is not a crop repair.",
        )
        dpi_x = st.number_input("Measured scan DPI X", min_value=1.0, value=300.0)
        dpi_y = st.number_input("Measured scan DPI Y", min_value=1.0, value=300.0)
        crop_x = st.number_input("Source left edge relative to paper left (mm)", value=0.0)
        crop_y = st.number_input("Source top edge relative to paper top (mm)", value=0.0)
        st.caption(
            "Crop offsets may be negative for scanner borders. Crop/DPI mode assumes an unskewed scan. "
            "Use corners for skew. PDF mode preserves CropBox offsets and undoes PDF display rotation."
        )
        profile_path = st.text_input("Optional measured printer profile JSON path", value="")

    kind_labels = {"auto": "Automatic", "scan": "Scanned page", "photo": "Photo of a page"}
    input_kind = st.selectbox(
        "Input type",
        list(kind_labels),
        format_func=kind_labels.get,
        help="Preview method only; use registration controls above to establish physical paper coordinates.",
    )
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
            help="Auto rotates Canon landscape A4 scans left. Override it after checking the preview.",
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
    if st.session_state.get("scan_source") != source:
        _clear_results()
        st.session_state["scan_source"] = source

    upload = None
    if source == _EMAIL_SOURCE:
        _email_scan_source(input_mode, orientation)
    else:
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

    prompt = st.text_input("Optional continuation prompt", placeholder="Leave blank for automatic continuation")
    live_gemini = st.toggle(
        "Live Gemini (paid API requests)",
        value=False,
        help="Uses the real Gemini image model and critic: up to 4 image and 4 critic requests per run. "
        "Validation, approval and export are unchanged.",
    )
    backend = GenerationBackend.GEMINI if live_gemini else GenerationBackend.OFFLINE
    live_ready = not live_gemini or gemini_api_key() is not None
    if live_gemini and not live_ready:
        st.warning(
            "Live Gemini needs an API key. Set GEMINI_API_KEY in the environment or in the local .env "
            "file (see .env.example), then restart the app. Generation is disabled until then."
        )
    elif live_gemini:
        st.caption("Live Gemini is on. Each run makes paid requests; the offline doubles stay available.")
    st.subheader("Allowed completion rectangle (millimetres from the top-left)")
    columns = st.columns(4)
    x_mm = columns[0].number_input("X", min_value=0.0, max_value=209.0, value=40.0, step=1.0)
    y_mm = columns[1].number_input("Y", min_value=0.0, max_value=296.0, value=60.0, step=1.0)
    width_mm = columns[2].number_input("Width", min_value=1.0, max_value=210.0, value=130.0, step=1.0)
    height_mm = columns[3].number_input("Height", min_value=1.0, max_value=297.0, value=180.0, step=1.0)

    explicit_registration = bool(
        source == _UPLOAD_SOURCE and confirmed_geometry and mode != "auto"
    )
    generation_key = (
        st.session_state.get("request_key"),
        input_kind,
        prompt,
        x_mm,
        y_mm,
        width_mm,
        height_mm,
        page_number,
        mode,
        confirmed_geometry,
        paper_width,
        paper_height,
        corners_text,
        nonuniform,
        dpi_x,
        dpi_y,
        crop_x,
        crop_y,
        profile_path,
        backend.value,
        bool(st.session_state.get("scanner_full_sheet")),
    )
    if st.session_state.get("generation_key") != generation_key:
        _clear_generation()
        st.session_state["generation_key"] = generation_key

    if source == _UPLOAD_SOURCE:
        prepare_label = "Generate two candidates" if explicit_registration else "Prepare normalized preview"
        if st.button(
            prepare_label,
            type="primary",
            disabled=upload is None or (explicit_registration and not live_ready),
        ):
            _clear_results()
            try:
                assert upload is not None
                upload_digest = sha256(upload.getvalue()).hexdigest()
                run_root = Path("artifacts/streamlit") / upload_digest[:12]
                suffix = Path(upload.name).suffix.lower()
                input_path = run_root / "input" / f"upload{suffix}"
                input_path.parent.mkdir(parents=True, exist_ok=True)
                input_path.write_bytes(upload.getvalue())
                with st.spinner("Loading and normalizing the selected input mode…"):
                    registration = RegistrationOptions(
                        mode=mode,
                        confirmed=confirmed_geometry,
                        page_size_mm=(paper_width, paper_height),
                        corners_px=json.loads(corners_text) if corners_text.strip() else None,
                        dpi_xy=(dpi_x, dpi_y),
                        origin_mm=(crop_x, crop_y),
                        allow_nonuniform_resize=nonuniform,
                    )
                    if explicit_registration:
                        normalization = _registered_preview(
                            input_path,
                            page_index=int(page_number) - 1,
                            input_kind=input_kind,
                            registration=registration,
                        )
                    else:
                        normalization = normalize_document_input(
                            input_path,
                            input_mode=input_mode,
                            orientation=orientation,
                            page_index=int(page_number) - 1,
                            layout=DEFAULT_LAYOUT,
                        )
                    st.session_state["input_normalization"] = normalization
                    st.session_state["upload_path"] = input_path
                    if explicit_registration:
                        allowed = rectangular_allowed_region(
                            x_mm=x_mm,
                            y_mm=y_mm,
                            width_mm=width_mm,
                            height_mm=height_mm,
                        )
                        st.session_state["mvp_run"] = run_offline_document(
                            input_path,
                            allowed,
                            input_path.parents[1] / "run",
                            user_prompt=prompt,
                            layout=DEFAULT_LAYOUT,
                            input_mode=input_mode,
                            orientation=orientation,
                            normalization_result=normalization,
                            normalization_confirmed=True,
                            input_kind=input_kind,
                            page_index=int(page_number) - 1,
                            registration=registration,
                            backend=backend,
                        )
                        st.session_state["generation_completed_for_preview"] = True
                        st.rerun()
            except LiveGenerationUnavailable as error:
                st.warning(str(error))
            except Exception as error:
                st.error(str(error))

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
    if not explicit_registration:
        confirmed_normalization = st.checkbox(
            "I confirm the orientation and placement in this normalized A4 preview look correct.",
            key="normalization_confirmed",
        )
        full_sheet = False
        scanner_profile = None
        if normalization.input_mode is InputMode.SCANNER_A4:
            full_sheet = st.checkbox(
                "This scan shows the complete A4 sheet (scanned on the glass at A4, nothing cut off).",
                key="scanner_full_sheet",
                help="Needed to print onto the original. It also needs the scanner placement measured once.",
            )
            scanner_profile = measured_scanner_profile()
            if full_sheet and scanner_profile is None:
                st.info(
                    "Printing onto the original stays locked until the scanner placement is measured once "
                    "(python -m printgpt_mvp.scanner_profile; see docs/mvp.md). You can still generate and preview."
                )
            elif full_sheet:
                st.caption(
                    f"Using the scanner profile measured {scanner_profile.get('measured_at')} "
                    f"(accuracy bound {scanner_profile['accuracy_bound_mm']:.2f} mm)."
                )
        if st.button(
            "Generate two candidates",
            type="primary",
            disabled=(
                not confirmed_normalization
                or not live_ready
                or bool(st.session_state.get("generation_completed_for_preview"))
            ),
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
                source_label = "live Gemini" if live_gemini else "offline"
                with st.spinner(f"Protecting existing ink and generating two {source_label} candidates…"):
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
                        input_kind=input_kind,
                        page_index=int(page_number)-1,
                        # A confirmed complete Scanner A4 sheet is registered from its
                        # normalized preview and the measured scanner profile instead.
                        registration=None if full_sheet else RegistrationOptions(
                            mode=mode, confirmed=confirmed_geometry,
                            page_size_mm=(paper_width, paper_height),
                            corners_px=json.loads(corners_text) if corners_text.strip() else None,
                            dpi_xy=(dpi_x, dpi_y), origin_mm=(crop_x, crop_y),
                            allow_nonuniform_resize=nonuniform,
                        ),
                        backend=backend,
                        scanner_full_sheet_confirmed=full_sheet,
                        scanner_profile=scanner_profile if full_sheet else None,
                    )
                    st.session_state["generation_completed_for_preview"] = True
                    st.rerun()
            except LiveGenerationUnavailable as error:
                st.warning(str(error))
            except Exception as error:
                st.error(str(error))

    run = st.session_state.get("mvp_run")
    if not isinstance(run, MvpRun):
        render_hardware_panel(st)
        return
    if run.alignment.get("alignment_method") == "scan_frame":
        st.info("The full scan was fitted for preview. Its image bounds have not been established as physical paper edges.")
    if run.alignment.get("alignment_method") == "page_boundary":
        st.info(
            "No usable calibration markers were found, so the page was aligned from its "
            "outer boundary. Check the preview carefully before approval."
        )
    for warning in run.alignment.get("warnings", ()):
        st.warning(warning)
    st.caption(
        f"Generator: `{run.generation.generator_model_id}` · critic: `{run.generation.critic_model_id}` · "
        f"requests: {run.generation.counters.generation_requests} image, {run.generation.counters.critic_requests} critic"
    )
    registered = run.alignment.get("registration", {}).get("export_allowed", False)
    if registered:
        st.info("Source geometry registered under the recorded assumptions. Printer accuracy is unmeasured until calibrated; use sacrificial sheets first.")
    elif run.alignment.get("registration", {}).get("status") == "awaiting_scanner_measurement":
        st.warning(
            "Preview is usable. Printing onto the original is locked until the scanner placement is measured "
            "once (python -m printgpt_mvp.scanner_profile; see docs/mvp.md)."
        )
    else:
        st.warning("Preview is usable, but export onto the original sheet is blocked until physical geometry is confirmed.")
    with st.expander("Source pixels and physical coordinate evidence"):
        source = load_document(run.input_path, page_index=run.alignment["page_index"])
        ruler = source.image.copy()
        height, width = ruler.shape[:2]
        step = max(100, round(max(width,height)/1000)*100)
        for x in range(0,width,step):
            cv2.line(ruler,(x,0),(x,20),(0,0,255),2)
            cv2.putText(ruler,str(x),(x+2,35),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        for y in range(step,height,step):
            cv2.line(ruler,(0,y),(20,y),(0,0,255),2)
            cv2.putText(ruler,str(y),(22,y),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
        st.image(cv2.cvtColor(ruler, cv2.COLOR_BGR2RGB), caption=f"Source: {width} x {height} pixels after EXIF/PDF display rotation. Red rulers label source pixels, not thumbnail pixels. Enter the physical sheet's four corners above.")
        st.json(run.alignment)
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
                right.image(candidate.attempt.preview_rgb, caption="Sanitized additions in paper coordinates — printer mechanics not simulated")
            metrics = candidate.attempt.validation.metrics if candidate.attempt.validation else {}
            normalization = candidate.attempt.normalization
            alignment = None if normalization is None else normalization.alignment
            if alignment is not None and alignment.applied:
                st.caption(
                    f"Model output was realigned onto the original: shifted "
                    f"{alignment.centre_shift_mm[0]:+.2f} / {alignment.centre_shift_mm[1]:+.2f} mm, "
                    f"scale {alignment.scale:.3f}, rotation {alignment.rotation_deg:+.2f}°."
                )
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
        if st.session_state.get("export_candidate") != choice:
            st.session_state.pop("approved_export", None)
            st.session_state["export_candidate"] = choice
        confirmed = st.checkbox("I approve this sanitized new-ink preview for A4 export.", key="approve_preview")
        if st.button("Approve and create toner-only files", disabled=not (confirmed and registered)):
            try:
                profile = json.loads(Path(profile_path).read_text(encoding="utf-8")) if profile_path.strip() else None
                st.session_state["approved_export"] = approve_and_export(run, choice, approved=True, printer_profile=profile)
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
    render_print_step(st, exported)
    render_hardware_panel(st)


if __name__ == "__main__":
    main()
