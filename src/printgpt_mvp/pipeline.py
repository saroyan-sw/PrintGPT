"""One dependable offline path from an aligned scan to an approved A4 overlay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
import json
from math import ceil, floor, isclose
import os
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
)
from printgpt_contracts import MaskBundle, MaskingConfig, ValidationConfig
from printgpt_alignment.detection import InputKind
from printgpt_alignment.document import load_document
from printgpt_alignment.registration import RegistrationOptions, register_document
from printgpt_generation import GenerationConfig, create_generation_request, run_generation, write_run_record
from printgpt_generation.doubles import (
    CriticScript,
    DeterministicCritic,
    DeterministicGenerator,
    GeneratorBehavior,
)
from printgpt_generation.graph import GraphDependencies
from printgpt_generation.models import AttemptRecord, RunStatus, require_binary_mask
from printgpt_generation.providers import FixedStepClock, SequentialIdProvider, SystemClock, UuidIdProvider
from printgpt_generation.requests import GenerationRunResult
from printgpt_generation.track_a import TrackAValidator
from printgpt_masking import build_masks

from .input_normalization import (
    InputMode,
    InputNormalizationResult,
    OrientationChoice,
    normalize_document_input,
)
from .print_approval import write_print_approval


PRINT_AT_ACTUAL_SIZE = (
    "Print this PDF at 100% or Actual Size. Disable Fit, Shrink, Scale to Fit, "
    "borderless expansion, and automatic rotation."
)
_MANIFEST_SCHEMA = "printgpt-mvp-run/2"


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
    normalization: InputNormalizationResult | None
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
    # Unregistered placement: only ever for a spare test sheet, never the original.
    spare_sheet_test: bool = False


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
    input_mode: InputMode | str | None = None,
    orientation: OrientationChoice | str = OrientationChoice.AUTO,
    input_kind: InputKind = "auto",
    registration: RegistrationOptions | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Canonicalize through an explicit input mode or physical registration."""
    path = Path(input_path)
    if input_mode is not None:
        normalization = normalize_document_input(
            path,
            page_index=page_index,
            layout=layout,
            input_mode=input_mode,
            orientation=orientation,
        )
        report = normalization.to_metadata()
        report["input_path"] = str(path)
        report["page_index"] = page_index
        return normalization.canonical_rgb, report

    source = load_document(path, page_index=page_index, dpi=layout.dpi)
    warped, report = register_document(source, layout=layout, input_kind=input_kind, options=registration)
    canonical = _rgb_from_alignment(warped)
    expected = (*layout.image_shape, 3)
    if canonical.shape != expected:
        raise ValueError(f"aligned page must have shape {expected}, got {canonical.shape}")
    report["input_path"] = str(path)
    report["page_index"] = page_index
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
    sx, sy = layout.raster_width_px / layout.page_width_mm, layout.raster_height_px / layout.page_height_mm
    x0 = max(0, floor(x_mm * sx))
    y0 = max(0, floor(y_mm * sy))
    x1 = min(layout.raster_width_px, ceil((x_mm + width_mm) * sx))
    y1 = min(layout.raster_height_px, ceil((y_mm + height_mm) * sy))
    mask = np.zeros(layout.image_shape, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


class GenerationBackend(StrEnum):
    """Which generator and critic run inside the unchanged LangGraph workflow."""

    OFFLINE = "offline"
    GEMINI = "gemini"


class LiveGenerationUnavailable(RuntimeError):
    """Live Gemini cannot start (no key or SDK); nothing has been written or requested."""


GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


def gemini_api_key(dotenv_path: Path | None = Path(".env")) -> str | None:
    """Return ``GEMINI_API_KEY`` from the environment, else from the local ``.env``."""
    key = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if not key and dotenv_path is not None:
        from printgpt_mvp.email_ingestion.settings import read_dotenv

        key = read_dotenv(dotenv_path).get(GEMINI_API_KEY_ENV, "").strip()
    return key or None


def gemini_dependencies(
    config: GenerationConfig,
    validation: ValidationConfig,
    *,
    settings: Any | None = None,
    client: Any | None = None,
) -> GraphDependencies:
    """Live dependencies: only the generator and critic change.

    ``TrackAValidator`` and the bounded graph stay exactly as offline. ``client``
    lets tests inject a mock-transport client; otherwise the key is required.
    """
    try:
        from printgpt_generation.gemini import (
            GeminiImageGenerator,
            GeminiSettings,
            GeminiVisualCritic,
            create_gemini_client,
        )
    except ImportError as error:
        raise LiveGenerationUnavailable(
            "Live Gemini needs the google-genai SDK: pip install -e \".[gemini]\""
        ) from error
    if client is None:
        key = gemini_api_key()
        if key is None:
            raise LiveGenerationUnavailable(
                f"Live Gemini needs an API key. Set {GEMINI_API_KEY_ENV} in the environment or in the "
                "local .env file (see .env.example). No request was made."
            )
        client = create_gemini_client(key)
    settings = settings or GeminiSettings()
    return GraphDependencies(
        generator=GeminiImageGenerator(client, settings),
        validator=TrackAValidator(validation),
        critic=GeminiVisualCritic(client, settings),
        clock=SystemClock(),
        ids=UuidIdProvider(),
        config=config,
    )


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
    normalization: InputNormalizationResult | None,
    *,
    approved: bool,
    exported_paths: dict[str, str | None],
    alignment: dict[str, Any] | None = None,
    printer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = selected.attempt.validation if selected is not None else None
    return {
        "schema_version": _MANIFEST_SCHEMA,
        "generation_run_id": generation.run_id,
        "input_normalization": None if normalization is None else normalization.to_metadata(),
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
        "registration": (alignment or {}).get("registration"),
        "coordinate_system": (alignment or {}).get("coordinate_system"),
        "printer": printer or {"status": "unmeasured", "correction_applied": False},
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
    input_mode: InputMode | str | None = None,
    orientation: OrientationChoice | str = OrientationChoice.AUTO,
    normalization_result: InputNormalizationResult | None = None,
    normalization_confirmed: bool | None = None,
    input_kind: InputKind = "auto",
    registration: RegistrationOptions | None = None,
    backend: GenerationBackend | str = GenerationBackend.OFFLINE,
    gemini_settings: Any | None = None,
    gemini_client: Any | None = None,
    scanner_full_sheet_confirmed: bool = False,
    scanner_profile: dict[str, Any] | None = None,
    loop_profile: dict[str, Any] | None = None,
) -> MvpRun:
    """Run alignment, masking, generation, validation, criticism, and selection.

    ``backend`` selects the deterministic offline doubles (default) or live
    Gemini. Live prerequisites are checked before anything is written, and no
    backend ever creates printable output: only ``approve_and_export`` does.

    A Scanner A4 page becomes printable onto the original only when the user
    confirms it shows the complete sheet *and* ``scanner_profile`` is a measured
    scan-to-paper profile (``printgpt_mvp.scanner_profile``); the page is then
    resampled into physical paper coordinates. Confirmation alone leaves export
    locked with status ``awaiting_scanner_measurement``.

    ``loop_profile`` (``printgpt_mvp.loop_calibration``) is the alternative to a
    scanner profile: it measures the whole scan -> print loop for one printer.
    The page then stays in scan coordinates and the export is corrected by the
    profile's loop correction.
    """
    if scanner_profile is not None and scanner_profile.get("status") != "measured":
        raise ValueError("scanner_profile must be a measured profile")
    if loop_profile is not None and loop_profile.get("status") != "measured":
        raise ValueError("loop_profile must be a measured, verified profile")
    backend = GenerationBackend(backend)
    config = GenerationConfig(layout=layout)
    if backend is GenerationBackend.GEMINI:
        dependencies = gemini_dependencies(
            config, validation_config, settings=gemini_settings, client=gemini_client
        )
    else:
        dependencies = _offline_dependencies(config, validation_config)
    normalization_requested = normalization_result is not None or input_mode is not None
    if normalization_requested and normalization_confirmed is not True:
        raise PermissionError("normalized page preview confirmation is required before generation")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    normalization = normalization_result
    if normalization_requested and normalization is None:
        assert input_mode is not None
        normalization = normalize_document_input(
            input_path,
            page_index=page_index,
            layout=layout,
            input_mode=input_mode,
            orientation=orientation,
        )
    if normalization is not None:
        if input_mode is not None and normalization.input_mode is not InputMode(input_mode):
            raise ValueError("normalization result input mode does not match the requested mode")
        if normalization.canonical_dimensions != layout.raster_size_px:
            raise ValueError("normalization result canonical dimensions do not match the layout")

    # Explicit registration establishes physical paper coordinates. Without it,
    # the confirmed scanner/marker preview remains the generation coordinate
    # source and scanner pages stay preview-only for physical export.
    if normalization is None or registration is not None:
        canonical, alignment = canonicalize_document(
            input_path,
            page_index=page_index,
            layout=layout,
            input_kind=input_kind,
            registration=registration,
        )
        if normalization is not None:
            metadata = normalization.to_metadata()
            alignment["input_normalization"] = metadata
            alignment["input_mode"] = normalization.input_mode.value
            alignment["marker_calibration_used"] = normalization.marker_calibration_used
            alignment["selected_rotation"] = normalization.selected_rotation.value
    else:
        canonical = normalization.canonical_rgb
        alignment = normalization.to_metadata()
        transform = np.asarray(normalization.applied_transform.matrix, dtype=np.float64)
        width, height = normalization.source_pixel_dimensions
        source_corners = np.asarray(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        observed = cv2.perspectiveTransform(source_corners[None], transform)[0]
        canonical_to_mm = np.asarray(
            (
                (layout.page_width_mm / layout.raster_width_px, 0, 0),
                (0, layout.page_height_mm / layout.raster_height_px, 0),
                (0, 0, 1),
            ),
            dtype=np.float64,
        )
        registered = normalization.marker_calibration_used
        mode, status = ("aruco", "registered") if registered else ("scanner_a4", "preview_only")
        evidence = "ArUco marker geometry" if registered else "scanner A4 normalization only"
        scale_and_origin = "established by markers" if registered else "unconfirmed"
        source_accuracy = None
        scanner_full_sheet = (
            not registered and scanner_full_sheet_confirmed and normalization.input_mode is InputMode.SCANNER_A4
        )
        if scanner_full_sheet:
            mode = "scanner_full_sheet"
            if scanner_profile is None and loop_profile is not None:
                registered, status = True, "registered"
                source_accuracy = float(loop_profile["accuracy_bound_mm"])
                evidence = (
                    f"complete sheet confirmed by the user; print-scan loop calibration for "
                    f"{loop_profile.get('printer_queue')} / {loop_profile.get('tray')} from "
                    f"{loop_profile.get('measured_at')} (verified error {source_accuracy:.2f} mm)"
                )
                scale_and_origin = "scan coordinates; export corrected by the loop calibration"
            elif scanner_profile is None:
                status = "awaiting_scanner_measurement"
                evidence = "complete sheet confirmed by the user; scanner placement not measured yet"
            else:
                from .scanner_profile import apply_scanner_profile, scan_to_paper_px

                canonical = apply_scanner_profile(canonical, scanner_profile, layout)
                transform = scan_to_paper_px(scanner_profile, layout) @ transform
                observed = cv2.perspectiveTransform(source_corners[None], transform)[0]
                registered, status = True, "registered"
                source_accuracy = float(scanner_profile["accuracy_bound_mm"])
                evidence = (
                    "complete sheet confirmed by the user; measured scanner profile from "
                    f"{scanner_profile.get('measured_at')} (holdout bound {source_accuracy:.2f} mm)"
                )
                scale_and_origin = "established by measured scanner profile"
        alignment.update(
            alignment_method="aruco" if mode == "aruco" else "scanner_a4",
            input_kind="auto" if mode == "aruco" else "scan",
            source_to_canonical=transform.tolist(),
            canonical_to_source=np.linalg.inv(transform).tolist(),
            observed_polygon_canonical_px=observed.tolist(),
            registration={
                "status": status,
                "export_allowed": registered,
                "evidence": evidence,
                "options": {"mode": mode},
                "page_size_mm": [layout.page_width_mm, layout.page_height_mm],
                "scale_and_origin": scale_and_origin,
                "measured_source_accuracy_mm": source_accuracy,
                "printer_accuracy": "unmeasured",
                **({"loop_correction_mm": loop_profile["correction_mm"],
                    "loop_calibration": {key: loop_profile.get(key) for key in
                                         ("printer_queue", "tray", "measured_at", "accuracy_bound_mm", "rounds")}}
                   if scanner_full_sheet and scanner_profile is None and loop_profile is not None else {}),
            },
            coordinate_system={
                "origin": "physical paper top-left",
                "axes": "x right, y down",
                "convention": "outer pixel edges; raster spans exact page dimensions",
                "canonical_px_to_mm": canonical_to_mm.tolist(),
                "source_to_mm": (canonical_to_mm @ transform).tolist(),
            },
        )
    alignment["input_path"] = str(Path(input_path))
    alignment["page_index"] = page_index
    _write_json(root / "alignment.json", alignment)
    effective_masking_config = masking_config
    marker_calibrated = bool(
        (normalization is not None and normalization.marker_calibration_used)
        or alignment.get("alignment_method") == "aruco"
        or (set(alignment.get("detected_ids", ())) & set(layout.expected_marker_ids))
    )
    if not marker_calibrated:
        effective_masking_config = replace(
            masking_config,
            exclude_calibration_markers=False,
            exclude_calibration_labels=False,
            layout_guide_clearance_mm=0.0,
        )
    if "observed_polygon_canonical_px" in alignment:
        # Never generate onto unknown/missing paper, including PDF CropBox regions.
        polygon = np.asarray(alignment["observed_polygon_canonical_px"], dtype=np.float32)
        footprint = np.zeros(layout.image_shape, dtype=np.uint8)
        cv2.fillConvexPoly(footprint, np.rint(polygon).astype(np.int32), 255)
        radius = ceil(effective_masking_config.effective_clearance_mm * max(
            layout.raster_width_px/layout.page_width_mm, layout.raster_height_px/layout.page_height_mm))
        if radius > 0:
            footprint = cv2.erode(footprint, np.ones((2*radius+1, 2*radius+1), np.uint8),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # Validate the caller's mask before intersection; don't coerce malformed input.
        require_binary_mask(allowed_region_mask, layout.image_shape, "allowed_region_mask")
        allowed_region_mask = cv2.bitwise_and(allowed_region_mask, footprint)
    masks = build_masks(
        canonical,
        allowed_region_mask,
        layout=layout,
        config=effective_masking_config,
    )
    request = create_generation_request(canonical, masks, user_prompt, config=config, seed=seed)
    generation = run_generation(request, dependencies)
    generation_record = write_run_record(generation, root / "generation")
    run = MvpRun(
        input_path=Path(input_path),
        output_dir=root,
        canonical_page_rgb=canonical,
        masks=masks,
        generation=generation,
        normalization=normalization,
        alignment=alignment,
        generation_record_path=generation_record,
        manifest_path=root / "mvp_manifest.json",
    )
    _write_json(
        run.manifest_path,
        _manifest_payload(
            generation,
            run.suggested_candidate,
            normalization,
            approved=False,
            exported_paths={"toner_png": None, "a4_pdf": None},
            alignment=alignment,
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
    correction_mm: np.ndarray | None = None,
) -> tuple[Path, Path]:
    if not (
        isclose(layout.page_width_mm, 210.0, abs_tol=1e-9)
        and isclose(layout.page_height_mm, 297.0, abs_tol=1e-9)
    ):
        raise ValueError("print export requires a 210 x 297 mm A4 layout")
    toner = _toner_raster(mask, layout)
    if correction_mm is not None:
        # Transform in physical mm, then rasterize once. Never modify the source
        # preview or its protection mask to compensate for printer mechanics.
        S = np.diag([layout.raster_width_px/210, layout.raster_height_px/297, 1.])
        H = S @ correction_mm @ np.linalg.inv(S)
        yy, xx = np.nonzero(mask)
        if len(xx):
            points = np.column_stack([xx, yy]).astype(np.float64)[None]
            shifted = cv2.perspectiveTransform(points, H)[0]
            if np.any(shifted < 0) or np.any(shifted >= [layout.raster_width_px-1, layout.raster_height_px-1]):
                raise ValueError("printer correction would clip new ink at the page edge")
        toner = cv2.warpPerspective(toner, H, (layout.raster_width_px, layout.raster_height_px),
                                    flags=cv2.INTER_NEAREST, borderValue=255)
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


def approve_and_export(run: MvpRun, attempt_id: str, *, approved: bool,
                       printer_profile: dict[str, Any] | None = None,
                       spare_sheet_test: bool = False) -> ApprovedExport:
    """Export only an explicitly approved, ranked, non-simulated sanitized mask.

    ``spare_sheet_test`` allows a page whose physical placement is not registered
    to be exported for a test print on a spare sheet. The export and its approval
    record are marked as such; every other check still applies.
    """
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
    registered = bool(run.alignment.get("registration", {}).get("export_allowed", False))
    if not registered and not spare_sheet_test:
        raise ValueError("preview only: confirm physical scale and paper origin before exporting onto the original sheet")
    spare_sheet_test = spare_sheet_test and not registered
    if spare_sheet_test and printer_profile is not None:
        raise ValueError("a printer correction needs registered source geometry; spare-sheet tests use none")
    layout = run.generation.config.layout
    correction = None
    loop_correction = (run.alignment.get("registration") or {}).get("loop_correction_mm")
    if loop_correction is not None and not spare_sheet_test:
        if printer_profile is not None:
            raise ValueError("the loop calibration already includes the printer; do not add a printer profile")
        correction = np.asarray(loop_correction, dtype=np.float64)
    if printer_profile is not None:
        from printgpt_alignment.printer_calibration import validated_correction
        correction = validated_correction(printer_profile)
        # A measured profile does not make an unmeasured source error disappear.
        source_bound = run.alignment["registration"].get("measured_source_accuracy_mm")
        from printgpt_alignment.printer_calibration import fit_printer, validate_printer
        printer_profile = validate_printer(fit_printer(printer_profile["calibration_record"]),
                                          printer_profile["validation_record"],
                                          tolerance_mm=printer_profile.get("tolerance_mm", .5))
        printer_bound = printer_profile["validation"]["max_error_mm"] + printer_profile["measurement_uncertainty_mm"]
        raster_bound = float(np.hypot(210/layout.raster_width_px, 297/layout.raster_height_px))
        total_bound = None if source_bound is None else source_bound + printer_bound + raster_bound
        if total_bound is not None and total_bound >= run.masks.config["effective_clearance_mm"]:
            raise ValueError("combined source/printer/raster error exceeds ink protection clearance; increase clearance and regenerate")
        printer_profile = dict(printer_profile, correction_applied=True,
                               observed_combined_error_bound_mm=total_bound,
                               protection_budget="unverified source accuracy" if total_bound is None else "within configured clearance on measured samples")
        yy, xx = np.nonzero(attempt.sanitized_new_ink_mask)
        if len(xx):
            intended = np.column_stack([xx*210/layout.raster_width_px, yy*297/layout.raster_height_px, np.ones(len(xx))])
            commanded = intended @ correction[:2].T
            left, top, right, bottom = printer_profile["safe_margins_mm"]
            if np.any(commanded < [left,top]) or np.any(commanded > [210-right,297-bottom]):
                raise ValueError("corrected ink falls inside measured printer margins; reduce the allowed region and regenerate")
    png_path, pdf_path = _write_toner_outputs(
        attempt.sanitized_new_ink_mask,
        run.output_dir / "print",
        layout,
        correction,
    )
    exported = {
        "toner_png": png_path.relative_to(run.output_dir).as_posix(),
        "a4_pdf": pdf_path.relative_to(run.output_dir).as_posix(),
    }
    alignment = run.alignment
    if spare_sheet_test:
        alignment = {**alignment, "registration": {
            **(alignment.get("registration") or {}),
            "status": "unregistered_spare_sheet_test", "export_allowed": False, "spare_sheet_test": True,
        }}
    _write_json(
        run.manifest_path,
        _manifest_payload(run.generation, selected, run.normalization,
                          approved=True, exported_paths=exported,
                          alignment=alignment, printer=printer_profile),
    )
    registration = alignment.get("registration") or {}
    write_print_approval(
        run_dir=run.output_dir,
        run_id=run.generation.run_id,
        candidate_id=selected.candidate_id,
        attempt_id=selected.attempt_id,
        toner_png=png_path,
        a4_pdf=pdf_path,
        validation={
            "accepted": attempt.validation.accepted,
            "simulated": attempt.validation.simulated,
            "reasons": list(attempt.validation.reasons),
        },
        input_mode={
            "input_kind": run.alignment.get("input_kind"),
            "alignment_method": run.alignment.get("alignment_method"),
            "registration_status": registration.get("status"),
            "registration_mode": (registration.get("options") or {}).get("mode"),
        },
        printer_correction_applied=correction is not None,
    )
    return ApprovedExport(selected.candidate_id, selected.attempt_id, png_path, pdf_path, run.manifest_path,
                          spare_sheet_test=spare_sheet_test)
