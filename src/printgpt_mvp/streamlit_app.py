"""Minimal approval UI for the deterministic offline PrintGPT MVP."""

from __future__ import annotations

from pathlib import Path
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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="PrintGPT MVP", layout="wide")
    st.title("PrintGPT — safe A4 completion preview")
    st.caption(
        "Offline deterministic demonstration. Nothing is exported until you explicitly "
        "approve a validated preview."
    )

    upload = st.file_uploader(
        "Upload a marked A4 scan or supported PDF",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "pdf"],
    )
    prompt = st.text_input("Optional continuation prompt", placeholder="Leave blank for automatic continuation")
    st.subheader("Allowed completion rectangle (millimetres from the top-left)")
    columns = st.columns(4)
    x_mm = columns[0].number_input("X", min_value=0.0, max_value=209.0, value=40.0, step=1.0)
    y_mm = columns[1].number_input("Y", min_value=0.0, max_value=296.0, value=60.0, step=1.0)
    width_mm = columns[2].number_input("Width", min_value=1.0, max_value=210.0, value=130.0, step=1.0)
    height_mm = columns[3].number_input("Height", min_value=1.0, max_value=297.0, value=180.0, step=1.0)

    if st.button("Generate two candidates", type="primary", disabled=upload is None):
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
                )
                st.session_state.pop("approved_export", None)
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
        confirmed = st.checkbox("I approve this sanitized new-ink preview for A4 export.")
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
