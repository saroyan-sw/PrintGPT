"""One dependable offline path from an aligned scan to an approved A4 overlay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, floor, isclose
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from printgpt_alignment import (
    A4CalibrationLayout,
    DEFAULT_LAYOUT,
    detect_and_rectify,
    load_document_page,
    mm_to_px,
)
from printgpt_contracts import MaskBundle, MaskingConfig, ValidationConfig
from printgpt_generation import GenerationConfig, create_generation_request, run_generation, write_run_record
from printgpt_generation.doubles import (
    CriticScript,
    DeterministicCritic,
    DeterministicGenerator,
    GeneratorBehavior,
)
from printgpt_generation.graph import GraphDependencies
from printgpt_generation.models import AttemptRecord, RunStatus, require_binary_mask
from printgpt_generation.providers import FixedStepClock, SequentialIdProvider
from printgpt_generation.requests import GenerationRunResult
from printgpt_generation.track_a import TrackAValidator
from printgpt_masking import build_masks


PRINT_AT_ACTUAL_SIZE = (
    "Print this PDF at 100% or Actual Size. Disable Fit, Shrink, Scale to Fit, "
    "borderless expansion, and automatic rotation."
)
_MANIFEST_SCHEMA = "printgpt-mvp-run/1"


@dataclass(frozen=True, slots=True)
class CandidateView:
    """One candidate's selected eligible attempt and user-facing evidence."""

    candidate_id: str
    attempt_id: str
    rank: int
    score: float
    attempt: AttemptRecord


@dataclass(frozen=True, slots=True, eq=False)
class MvpRun:
    """An unapproved integrated run. No printable file exists yet."""

    input_path: Path
    output_dir: Path
    canonical_page_rgb: np.ndarray
    masks: MaskBundle
    generation: GenerationRunResult
    alignment: dict[str, Any]
    generation_record_path: Path
    manifest_path: Path

    @property
    def candidates(self) -> tuple[CandidateView, ...]:
        return tuple(
            CandidateView(
                candidate_id=item.candidate_id,
                attempt_id=item.attempt_id,
                rank=item.rank,
                score=item.score,
                attempt=self.generation.attempt(item.attempt_id),
            )
            for item in self.generation.outcome.ranked
        )

    @property
    def suggested_candidate(self) -> CandidateView | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True, slots=True)
class ApprovedExport:
    candidate_id: str
    attempt_id: str
    toner_png_path: Path
    a4_pdf_path: Path
    manifest_path: Path
    print_instruction: str = PRINT_AT_ACTUAL_SIZE


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rgb_from_alignment(warped: np.ndarray) -> np.ndarray:
    if warped.ndim == 2:
        rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
    elif warped.ndim == 3 and warped.shape[2] == 3:
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    elif warped.ndim == 3 and warped.shape[2] == 4:
        rgb = cv2.cvtColor(warped, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"unsupported aligned image shape: {warped.shape}")
    if rgb.dtype != np.uint8:
        raise ValueError(f"aligned image must have dtype uint8, got {rgb.dtype}")
    return np.ascontiguousarray(rgb)


def canonicalize_document(
    input_path: str | Path,
    *,
    page_index: int = 0,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one supported image/PDF page, rectify it, and convert BGR to RGB once."""
    path = Path(input_path)
    source = load_document_page(path, page_index=page_index, dpi=layout.dpi)
    detection = detect_and_rectify(source, layout)
    if not detection.accepted or detection.warped is None:
        raise ValueError(f"page alignment rejected: {detection.reason}")
    canonical = _rgb_from_alignment(detection.warped)
    expected = (*layout.image_shape, 3)
    if canonical.shape != expected:
        raise ValueError(f"aligned page must have shape {expected}, got {canonical.shape}")
    report = detection.to_dict()
    report["input_path"] = str(path)
    report["page_index"] = page_index
    report["canonical_colour_space"] = "RGB"
    return canonical, report


def rectangular_allowed_region(
    *,
    x_mm: float,
    y_mm: float,
    width_mm: float,
    height_mm: float,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
) -> np.ndarray:
    """Build an explicit 0/255 completion rectangle in canonical coordinates."""
    values = (x_mm, y_mm, width_mm, height_mm)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("allowed-region dimensions must be finite")
    if x_mm < 0 or y_mm < 0 or width_mm <= 0 or height_mm <= 0:
        raise ValueError("allowed region must have a positive size and non-negative origin")
    if x_mm + width_mm > layout.page_width_mm or y_mm + height_mm > layout.page_height_mm:
        raise ValueError("allowed region must stay inside the A4 page")
    x0 = max(0, floor(mm_to_px(x_mm, layout.dpi)))
    y0 = max(0, floor(mm_to_px(y_mm, layout.dpi)))
    x1 = min(layout.raster_width_px, ceil(mm_to_px(x_mm + width_mm, layout.dpi)))
    y1 = min(layout.raster_height_px, ceil(mm_to_px(y_mm + height_mm, layout.dpi)))
    mask = np.zeros(layout.image_shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def _offline_dependencies(config: GenerationConfig, validation: ValidationConfig) -> GraphDependencies:
    return GraphDependencies(
        generator=DeterministicGenerator({(1, 0): GeneratorBehavior.OUTSIDE_REGION}),
        validator=TrackAValidator(validation),
        critic=DeterministicCritic(
            {
                (0, 0): CriticScript.accept(7.5),
                (1, 1): CriticScript.accept(8.5),
            }
        ),
        clock=FixedStepClock(),
        ids=SequentialIdProvider(),
        config=config,
    )


def _manifest_payload(
    generation: GenerationRunResult,
    selected: CandidateView | None,
    *,
    approved: bool,
    exported_paths: dict[str, str | None],
) -> dict[str, Any]:
    validation = selected.attempt.validation if selected is not None else None
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "generation_run_id": generation.run_id,
        "status": generation.outcome.status.value,
        "selected_candidate": None
        if selected is None
        else {
            "candidate_id": selected.candidate_id,
            "attempt_id": selected.attempt_id,
            "rank": selected.rank,
            "critic_score": selected.score,
        },
        "validation": None
        if validation is None
        else {
            "accepted": validation.accepted,
            "reasons": list(validation.reasons),
            "metrics": validation.metrics,
            "simulated": validation.simulated,
        },
        "approved": approved,
        "exported_paths": exported_paths,
        "print_instruction": PRINT_AT_ACTUAL_SIZE,
    }


def run_offline_document(
    input_path: str | Path,
    allowed_region_mask: np.ndarray,
    output_dir: str | Path,
    *,
    user_prompt: str | None = None,
    page_index: int = 0,
    seed: int = 0,
    layout: A4CalibrationLayout = DEFAULT_LAYOUT,
    masking_config: MaskingConfig = MaskingConfig(),
    validation_config: ValidationConfig = ValidationConfig(),
) -> MvpRun:
    """Run alignment, masking, generation, validation, criticism, and selection offline."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canonical, alignment = canonicalize_document(input_path, page_index=page_index, layout=layout)
    masks = build_masks(canonical, allowed_region_mask, layout=layout, config=masking_config)
    config = GenerationConfig(layout=layout)
    request = create_generation_request(canonical, masks, user_prompt, config=config, seed=seed)
    generation = run_generation(request, _offline_dependencies(config, validation_config))
    generation_record = write_run_record(generation, root / "generation")
    run = MvpRun(
        input_path=Path(input_path),
        output_dir=root,
        canonical_page_rgb=canonical,
        masks=masks,
        generation=generation,
        alignment=alignment,
        generation_record_path=generation_record,
        manifest_path=root / "mvp_manifest.json",
    )
    _write_json(
        run.manifest_path,
        _manifest_payload(
            generation,
            run.suggested_candidate,
            approved=False,
            exported_paths={"toner_png": None, "a4_pdf": None},
        ),
    )
    return run


def _candidate_for_approval(run: MvpRun, attempt_id: str) -> CandidateView:
    for candidate in run.candidates:
        if candidate.attempt_id == attempt_id:
            return candidate
    raise ValueError("only a ranked, validated candidate can be approved")


def _toner_raster(mask: np.ndarray, layout: A4CalibrationLayout) -> np.ndarray:
    require_binary_mask(mask, layout.image_shape, "sanitized_new_ink_mask")
    raster = np.full(layout.image_shape, 255, dtype=np.uint8)
    raster[mask == 255] = 0
    return raster


def _write_toner_outputs(
    mask: np.ndarray,
    output_dir: Path,
    layout: A4CalibrationLayout,
) -> tuple[Path, Path]:
    if not (
        isclose(layout.page_width_mm, 210.0, abs_tol=1e-9)
        and isclose(layout.page_height_mm, 297.0, abs_tol=1e-9)
    ):
        raise ValueError("print export requires a 210 x 297 mm A4 layout")
    toner = _toner_raster(mask, layout)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "approved_toner_only.png"
    Image.fromarray(toner, mode="L").save(png_path, dpi=(layout.dpi, layout.dpi))

    pdf_path = output_dir / "approved_a4_actual_size.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(210.0 * mm, 297.0 * mm), pageCompression=1)
    pdf.drawImage(
        ImageReader(Image.fromarray(toner, mode="L")),
        0,
        0,
        width=210.0 * mm,
        height=297.0 * mm,
        preserveAspectRatio=False,
        anchor="sw",
    )
    pdf.showPage()
    pdf.save()
    return png_path, pdf_path


def approve_and_export(run: MvpRun, attempt_id: str, *, approved: bool) -> ApprovedExport:
    """Export only an explicitly approved, ranked, non-simulated sanitized mask."""
    if not approved:
        raise PermissionError("explicit user approval is required before export")
    if run.generation.simulated_validation:
        raise ValueError("simulated validation is never print-ready")
    if run.generation.outcome.status is not RunStatus.CANDIDATES_READY:
        raise ValueError("the run has no candidate eligible for approval")
    selected = _candidate_for_approval(run, attempt_id)
    attempt = selected.attempt
    if attempt.validation is None or not attempt.validation.accepted or attempt.validation.simulated:
        raise ValueError("candidate validation is missing, rejected, or simulated")
    if attempt.sanitized_new_ink_mask is None:
        raise ValueError("candidate has no sanitized new-ink mask")
    layout = run.generation.config.layout
    png_path, pdf_path = _write_toner_outputs(
        attempt.sanitized_new_ink_mask,
        run.output_dir / "print",
        layout,
    )
    exported = {
        "toner_png": png_path.relative_to(run.output_dir).as_posix(),
        "a4_pdf": pdf_path.relative_to(run.output_dir).as_posix(),
    }
    _write_json(
        run.manifest_path,
        _manifest_payload(run.generation, selected, approved=True, exported_paths=exported),
    )
    return ApprovedExport(selected.candidate_id, selected.attempt_id, png_path, pdf_path, run.manifest_path)
