"""Minimal approval UI for the deterministic offline PrintGPT MVP."""

from __future__ import annotations

from pathlib import Path
from hashlib import sha256
from uuid import uuid4

from printgpt_alignment import DEFAULT_LAYOUT

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
        help="Scanned page uses the entire image, including scanner borders. Photo finds the paper edges.",
    )
    prompt = st.text_input("Optional continuation prompt", placeholder="Leave blank for automatic continuation")
    st.subheader("Allowed completion rectangle (millimetres from the top-left)")
    columns = st.columns(4)
    x_mm = columns[0].number_input("X", min_value=0.0, max_value=209.0, value=40.0, step=1.0)
    y_mm = columns[1].number_input("Y", min_value=0.0, max_value=296.0, value=60.0, step=1.0)
    width_mm = columns[2].number_input("Width", min_value=1.0, max_value=210.0, value=130.0, step=1.0)
    height_mm = columns[3].number_input("Height", min_value=1.0, max_value=297.0, value=180.0, step=1.0)

    request_key = (
        sha256(upload.getvalue()).hexdigest() if upload is not None else None,
        input_kind, prompt, x_mm, y_mm, width_mm, height_mm,
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
        return
    if run.alignment.get("alignment_method") == "scan_frame":
        st.info("The full scanned image was used as the page. No markers or surrounding background are needed.")
    if run.alignment.get("alignment_method") == "page_boundary":
        st.info(
            "No usable calibration markers were found, so the page was aligned from its "
            "outer boundary. Check the preview carefully before approval."
        )
    for warning in run.alignment.get("warnings", ()):
        st.warning(warning)
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
