"""Run the agentic generation workflow on one real scan and record live-verification evidence.

Rehearse for free with the offline doubles (default), then run live:

    .venv\\Scripts\\python.exe scripts\\live_gemini_run.py <scan.pdf> --region 20 258 170 30
    .venv\\Scripts\\python.exe scripts\\live_gemini_run.py <scan.pdf> --region 20 258 170 30 --backend gemini --confirm-paid

A live run makes paid requests: at most 4 image and 4 critic requests (plus
bounded transport retries). The key comes from ``GEMINI_API_KEY`` or the
``--dotenv`` file and is never printed or written. Nothing printable is
created; outputs are diagnostics and sanitized previews under ``--out``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from importlib import metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

# Use this checkout's sources even when the shared venv has another checkout installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from printgpt_alignment import DEFAULT_LAYOUT  # noqa: E402
from printgpt_generation.framing import expand, mask_bounds  # noqa: E402
from printgpt_generation.models import AttemptRecord, ProviderFailureKind  # noqa: E402
from printgpt_mvp.input_normalization import InputMode, OrientationChoice, normalize_document_input  # noqa: E402
from printgpt_mvp.pipeline import (  # noqa: E402
    GEMINI_API_KEY_ENV,
    GenerationBackend,
    LiveGenerationUnavailable,
    MvpRun,
    gemini_api_key,
    rectangular_allowed_region,
    run_offline_document,
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="scan PDF or image (Scanner A4 mode)")
    parser.add_argument("--region", nargs=4, type=float, metavar=("X", "Y", "W", "H"), required=True,
                        help="allowed region in mm from the page's top-left")
    parser.add_argument("--prompt", default="", help="optional continuation prompt (blank = automatic)")
    parser.add_argument("--backend", choices=[item.value for item in GenerationBackend], default="offline")
    parser.add_argument("--confirm-paid", action="store_true", help="required for --backend gemini")
    parser.add_argument("--framing", choices=["crop", "full_page"], default="crop")
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--critic-model", default=None)
    parser.add_argument("--orientation", choices=[item.value for item in OrientationChoice], default="auto")
    parser.add_argument("--page", type=int, default=1, help="1-based PDF page")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dotenv", type=Path, default=Path(".env"), help="file that may hold GEMINI_API_KEY")
    parser.add_argument("--out", type=Path, default=None, help="default: artifacts/live_gemini/<backend>-<UTC time>")
    return parser.parse_args(argv)


def _sdk_version() -> str | None:
    try:
        return metadata.version("google-genai")
    except metadata.PackageNotFoundError:
        return None


def _critic_record(attempt: AttemptRecord) -> dict[str, Any] | None:
    critic = attempt.critic
    if critic is None:
        return None
    verdict = critic.verdict
    malformed = critic.failure is not None and critic.failure.kind is ProviderFailureKind.MALFORMED_OUTPUT
    return {
        "model_id": critic.model_id,
        "schema_compliant": verdict is not None,
        "malformed_output": malformed,
        "failure": None if critic.failure is None else critic.failure.message,
        "provider_attempts": critic.provider_attempts,
        "latency_s": round(critic.latency_s, 2),
        # By construction the critic sees the sanitized preview and its region close-up, never raw output.
        "inputs": ["original_page", "region_guidance", "sanitized_preview", "sanitized_region_close_up"],
        "score": None if verdict is None else verdict.scores.score,
        "sub_scores": None if verdict is None else {
            "boundary_continuity": verdict.scores.boundary_continuity_score,
            "style_match": verdict.scores.style_match_score,
            "geometry_plausibility": verdict.scores.geometry_plausibility_score,
        },
        "recommendation": None if verdict is None else verdict.recommendation,
        "assessment": None if verdict is None else verdict.assessment.value,
        "problems": [] if verdict is None else list(verdict.problems),
        "usage": {"input_tokens": critic.usage.input_tokens, "output_tokens": critic.usage.output_tokens},
    }


def _attempt_record(attempt: AttemptRecord) -> dict[str, Any]:
    generation = attempt.generation
    output = generation.output
    normalization = attempt.normalization
    registration = None if normalization is None else normalization.registration
    extraction = None if normalization is None else normalization.extraction
    return {
        "attempt_id": attempt.attempt_id,
        "kind": attempt.kind.value,
        "status": attempt.status.value,
        "seed": attempt.seed,
        "generation": {
            "model_id": None if output is None else output.model_id,
            "provider_attempts": generation.provider_attempts,
            "latency_s": round(generation.latency_s, 2),
            "failure": None if generation.failure is None else f"{generation.failure.kind.value}: {generation.failure.message}",
            "raw_shape": None if output is None else list(output.image.shape),
            "canvas_box_px": None if output is None or output.canvas_box is None else list(output.canvas_box),
            "usage": None if output is None else {"input_tokens": output.usage.input_tokens,
                                                  "output_tokens": output.usage.output_tokens},
        },
        "normalization": None if normalization is None else {
            "aspect_ratio_error": round(normalization.aspect_ratio_error, 5),
            "failure_reasons": list(normalization.failure_reasons),
        },
        "registration": None if registration is None else {
            "verified": registration.verified,
            "estimated_shift_mm": round(registration.estimated_shift_mm, 3),
            "source_retention": round(registration.source_retention, 4),
            "source_ink_pixels": registration.source_ink_pixels,
            "phase_response": round(registration.phase_response, 4),
        },
        "extraction": None if extraction is None else {
            "candidate_ink_pixels": extraction.candidate_ink_pixels,
            "discarded_source_redraw_pixels": extraction.discarded_source_redraw_pixels,
            "discarded_noise_pixels": extraction.discarded_noise_pixels,
            "discarded_coloured_pixels": extraction.discarded_coloured_pixels,
            "new_ink_pixels": extraction.new_ink_pixels,
        },
        "validation": None if attempt.validation is None else {
            "accepted": attempt.validation.accepted,
            "reasons": list(attempt.validation.reasons),
            "final_ink_count": attempt.validation.metrics.get("final_ink_count"),
            "removed_fraction": attempt.validation.metrics.get("removed_fraction"),
        },
        "critic": _critic_record(attempt),
        "revision_decision": None if attempt.revision_decision is None else {
            "kind": attempt.revision_decision.kind.value,
            "reasons": list(attempt.revision_decision.reasons),
        },
    }


def _label(image: np.ndarray, text: str) -> np.ndarray:
    labelled = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(labelled, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return labelled


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, int(round(image.shape[0] * width / image.shape[1])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _contact_sheet(run: MvpRun, path: Path) -> None:
    """One row per attempt: raw provider output (diagnostic) and sanitized region close-up."""
    region = mask_bounds(run.masks.allowed_region_mask)
    rows = []
    for attempt in run.generation.attempts:
        raw = attempt.generation.output.image if attempt.generation.output is not None else None
        raw_tile = (_fit(raw[:, :, :3], 700) if raw is not None and raw.ndim == 3
                    else np.full((300, 700, 3), 230, np.uint8))
        preview = attempt.preview_rgb
        if preview is not None and region is not None:
            close_up = expand(region, 15 * DEFAULT_LAYOUT.dpi / 25.4, preview.shape).crop(preview)
            preview_tile = _fit(close_up, 1100)
        else:
            preview_tile = np.full((300, 1100, 3), 230, np.uint8)
        tiles = [_label(raw_tile, f"{attempt.attempt_id} raw (not printable)"),
                 _label(preview_tile, f"{attempt.status.value}: sanitized preview")]
        height = max(tile.shape[0] for tile in tiles)
        rows.append(np.hstack([
            cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for tile in tiles
        ]))
    if not rows:
        return
    width = max(row.shape[1] for row in rows)
    sheet = np.vstack([cv2.copyMakeBorder(row, 0, 12, 0, width - row.shape[1], cv2.BORDER_CONSTANT,
                                          value=(255, 255, 255)) for row in rows])
    cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    backend = GenerationBackend(args.backend)
    settings = None
    if backend is GenerationBackend.GEMINI:
        if not args.confirm_paid:
            print("Live Gemini makes paid requests (up to 4 image + 4 critic). Re-run with --confirm-paid.")
            return 2
        key = gemini_api_key(args.dotenv)
        if key is None:
            print(f"No {GEMINI_API_KEY_ENV} found in the environment or {args.dotenv}. No request was made.")
            return 2
        os.environ[GEMINI_API_KEY_ENV] = key
        from printgpt_generation.gemini import GeminiSettings

        overrides = {"framing": args.framing}
        if args.image_model:
            overrides["image_model"] = args.image_model
        if args.critic_model:
            overrides["critic_model"] = args.critic_model
        settings = GeminiSettings(**overrides)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path("artifacts/live_gemini") / f"{backend.value}-{stamp}"
    x, y, width, height = args.region
    normalization = normalize_document_input(
        args.input, input_mode=InputMode.SCANNER_A4, orientation=args.orientation,
        page_index=args.page - 1, layout=DEFAULT_LAYOUT,
    )
    try:
        run = run_offline_document(
            args.input,
            rectangular_allowed_region(x_mm=x, y_mm=y, width_mm=width, height_mm=height, layout=DEFAULT_LAYOUT),
            out,
            user_prompt=args.prompt,
            seed=args.seed,
            layout=DEFAULT_LAYOUT,
            input_mode=InputMode.SCANNER_A4,
            orientation=args.orientation,
            normalization_result=normalization,
            normalization_confirmed=True,  # Scanner A4 preview of this scan was confirmed by the user.
            page_index=args.page - 1,
            backend=backend,
            gemini_settings=settings,
        )
    except LiveGenerationUnavailable as error:
        print(error)
        return 2

    generation = run.generation
    report = {
        "schema": "printgpt-live-gemini-report/1",
        "created_at": stamp,
        "backend": backend.value,
        "input": str(args.input),
        "allowed_region_mm": {"x": x, "y": y, "width": width, "height": height},
        "framing": args.framing if backend is GenerationBackend.GEMINI else "offline_double",
        "generator_model_id": generation.generator_model_id,
        "critic_model_id": generation.critic_model_id,
        "google_genai_version": _sdk_version(),
        "status": generation.outcome.status.value,
        "outcome_reasons": list(generation.outcome.reasons),
        "counters": {
            "generation_requests": generation.counters.generation_requests,
            "critic_requests": generation.counters.critic_requests,
            "provider_attempts_including_retries": generation.counters.provider_attempts,
        },
        "ranked": [{"rank": item.rank, "attempt_id": item.attempt_id, "score": item.score} for item in generation.outcome.ranked],
        "attempts": [_attempt_record(attempt) for attempt in generation.attempts],
        "registration_status": run.alignment.get("registration", {}).get("status"),
        "printable_output_created": (out / "print").exists(),
        "generation_record": str(run.generation_record_path),
    }
    report_path = out / "live_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _contact_sheet(run, out / "contact_sheet.png")

    print(f"{backend.value}: {report['status']} | generator {report['generator_model_id']} | "
          f"critic {report['critic_model_id']} | SDK {report['google_genai_version']}")
    print(f"requests: {report['counters']}")
    for item in report["attempts"]:
        registration = item["registration"] or {}
        critic = item["critic"] or {}
        print(f"  {item['attempt_id']:<24} {item['status']:<20} shift {registration.get('estimated_shift_mm')} mm | "
              f"retention {registration.get('source_retention')} | score {critic.get('score')} | "
              f"failure {item['generation']['failure']}")
    print(f"report: {report_path}\ncontact sheet: {out / 'contact_sheet.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
