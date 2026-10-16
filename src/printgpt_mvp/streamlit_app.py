"""Minimal approval UI for the deterministic offline PrintGPT MVP."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from uuid import uuid4
import json
import cv2

from printgpt_alignment import DEFAULT_LAYOUT
from printgpt_alignment.document import load_document
from printgpt_alignment.registration import RegistrationOptions

from printgpt_mvp.hardware_panel import render_hardware_panel
from printgpt_mvp.pipeline import (
    PRINT_AT_ACTUAL_SIZE,
    MvpRun,
    approve_and_export,
    rectangular_allowed_region,
    run_offline_document,
)


def _candidate_label(run: MvpRun, attempt_id: str) -> str:
    candidate = next(item for item in run.candidates if item.attempt_id == attempt_id)
    return f"Candidate {candidate.candidate_id} — score {candidate.score:.1f}"


def _clear_results() -> None:
    import streamlit as st

    for key in ("mvp_run", "approved_export", "approve_preview"):
        st.session_state.pop(key, None)


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="PrintGPT MVP", layout="wide")
    st.title("PrintGPT — safe A4 completion preview")
    st.caption(
        "Upload a scanned document or a photo of a page. Markers are optional. Output uses A4. "
        "This is the offline demonstration with two generated sample additions. "
        "Nothing is exported until you explicitly approve a validated preview."
    )

    upload = st.file_uploader(
        "Upload a scan or PDF (markers optional)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
    )
    kind_labels = {"auto": "Automatic", "scan": "Scanned page", "photo": "Photo of a page"}
    input_kind = st.selectbox(
        "Input type",
        list(kind_labels),
        format_func=kind_labels.get,
        help="These are preview methods, not proof of physical paper position. Use registration controls below for export.",
    )
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
        confirmed_geometry = st.selectbox("Geometry confirmation", [False, True],
            format_func=lambda v: "I verified the paper size, orientation, scale and origin" if v else "Not verified — preview only")
        paper_width = st.number_input("Actual paper width (mm)", min_value=1., value=210.)
        paper_height = st.number_input("Actual paper height (mm)", min_value=1., value=297.)
        corners_text = st.text_area("Paper corners in source pixels: [[TL x,y],[TR x,y],[BR x,y],[BL x,y]]", value="",
            help="Use the true paper edges, not the text box. Order by the physical sheet's reading orientation. Inspect source pixels in the geometry panel after a preview run.")
        nonuniform = st.selectbox("Complete sheet was nonuniformly resized", [False, True],
            help="Only enable if the whole sheet is present and has been stretched. This is NOT a way to repair a crop.")
        dpi_x = st.number_input("Measured scan DPI X", min_value=1., value=300.)
        dpi_y = st.number_input("Measured scan DPI Y", min_value=1., value=300.)
        crop_x = st.number_input("Source left edge relative to paper left (mm)", value=0.)
        crop_y = st.number_input("Source top edge relative to paper top (mm)", value=0.)
        st.caption("Crop offsets may be negative for scanner borders. Crop/DPI mode assumes an unskewed scan. Use corners for skew. PDF mode preserves CropBox offsets and undoes PDF display rotation; embedded scan placement still needs checking.")
        profile_path = st.text_input("Optional measured printer profile JSON path", value="")
    prompt = st.text_input("Optional continuation prompt", placeholder="Leave blank for automatic continuation")
    st.subheader("Allowed completion rectangle (millimetres from the top-left)")
    columns = st.columns(4)
    x_mm = columns[0].number_input("X", min_value=0.0, max_value=209.0, value=40.0, step=1.0)
    y_mm = columns[1].number_input("Y", min_value=0.0, max_value=296.0, value=60.0, step=1.0)
    width_mm = columns[2].number_input("Width", min_value=1.0, max_value=210.0, value=130.0, step=1.0)
    height_mm = columns[3].number_input("Height", min_value=1.0, max_value=297.0, value=180.0, step=1.0)

    request_key = (
        sha256(upload.getvalue()).hexdigest() if upload is not None else None,
        input_kind, prompt, x_mm, y_mm, width_mm, height_mm, page_number,
        mode, confirmed_geometry, paper_width, paper_height, corners_text,
        nonuniform, dpi_x, dpi_y, crop_x, crop_y, profile_path,
    )
    if st.session_state.get("request_key") != request_key:
        _clear_results()
        st.session_state["request_key"] = request_key

    if st.button("Generate two candidates", type="primary", disabled=upload is None):
        _clear_results()
        try:
            allowed = rectangular_allowed_region(
                x_mm=x_mm,
                y_mm=y_mm,
                width_mm=width_mm,
                height_mm=height_mm,
            )
            run_root = Path("artifacts/streamlit") / uuid4().hex[:12]
            suffix = Path(upload.name).suffix.lower() if upload is not None else ".png"
            input_path = run_root / "input" / f"upload{suffix}"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(upload.getvalue())
            with st.spinner("Aligning, protecting existing ink, and generating two candidates…"):
                st.session_state["mvp_run"] = run_offline_document(
                    input_path,
                    allowed,
                    run_root / "run",
                    user_prompt=prompt,
                    layout=DEFAULT_LAYOUT,
                    input_kind=input_kind,
                    page_index=int(page_number)-1,
                    registration=RegistrationOptions(
                        mode=mode, confirmed=confirmed_geometry,
                        page_size_mm=(paper_width, paper_height),
                        corners_px=json.loads(corners_text) if corners_text.strip() else None,
                        dpi_xy=(dpi_x, dpi_y), origin_mm=(crop_x, crop_y),
                        allow_nonuniform_resize=nonuniform,
                    ),
                )
                st.session_state.pop("approved_export", None)
        except Exception as error:
            message = str(error)
            if "scan again" in message.lower():
                st.error(f"Scan again required — {message}")
            else:
                st.error(message)

    run = st.session_state.get("mvp_run")
    if not isinstance(run, MvpRun):
        render_hardware_panel(st, None)
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
    registered = run.alignment.get("registration", {}).get("export_allowed", False)
    if registered:
        st.info("Source geometry registered under the recorded assumptions. Printer accuracy is unmeasured until calibrated; use sacrificial sheets first.")
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
    render_hardware_panel(st, exported)


if __name__ == "__main__":
    main()
