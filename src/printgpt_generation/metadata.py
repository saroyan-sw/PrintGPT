"""JSON-safe run records and artifact persistence.

The record holds compact decisions, metrics, and relative artifact paths; images
and masks are written as separate PNG files. Credentials are never part of any
object reachable from a run result, so nothing here can serialize them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue

from .models import AttemptRecord, BinaryMask, RGBImage
from .requests import GenerationRunResult


SCHEMA_VERSION = "printgpt-generation-run/1"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AttemptMetadata(_Record):
    attempt_id: str
    candidate_id: str
    attempt_index: int
    kind: str
    status: str
    revision_of: str | None
    seed: int
    prompt: str
    model_id: str | None
    generation: dict[str, JsonValue]
    normalization: dict[str, JsonValue] | None
    validation: dict[str, JsonValue] | None
    critic: dict[str, JsonValue] | None
    revision_decision: dict[str, JsonValue] | None
    artifacts: dict[str, str]


class RunMetadata(_Record):
    schema_version: str
    run_id: str
    status: str
    requires_user_approval: bool
    approved: bool
    started_at: datetime
    finished_at: datetime
    duration_s: float
    user_prompt: str | None
    prompt_used: str
    automatic_prompt: bool
    seed: int
    generator_model_id: str
    critic_model_id: str
    simulated_validation: bool
    config: dict[str, JsonValue]
    counters: dict[str, int]
    revision_counts: dict[str, int]
    ranking: list[dict[str, JsonValue]]
    selections: list[dict[str, JsonValue]]
    outcome_reasons: list[str]
    attempts: list[AttemptMetadata]
    artifacts: dict[str, str]


def _json_safe(value: object) -> JsonValue:
    """Convert dataclass/enum/tuple/NumPy values into plain JSON values."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"value of type {type(value).__name__} is not JSON-safe")


def _dataclass_dict(value: object | None) -> dict[str, JsonValue] | None:
    if value is None:
        return None
    result = _json_safe(asdict(value))  # type: ignore[call-overload]
    assert isinstance(result, dict)
    return result


def _attempt_metadata(attempt: AttemptRecord, artifacts: Mapping[str, str]) -> AttemptMetadata:
    generation = attempt.generation
    output = generation.output
    generation_record: dict[str, JsonValue] = {
        "provider_attempts": generation.provider_attempts,
        "latency_s": generation.latency_s,
        "failure": _dataclass_dict(generation.failure),
        "raw_shape": list(output.image.shape) if output is not None else None,
        "has_additions_layer": output is not None and output.additions_mask is not None,
        "canvas_box": list(output.canvas_box) if output is not None and output.canvas_box else None,
        "usage": _dataclass_dict(output.usage) if output is not None else None,
    }
    critic = _dataclass_dict(attempt.critic)
    return AttemptMetadata(
        attempt_id=attempt.attempt_id,
        candidate_id=attempt.candidate_id,
        attempt_index=attempt.attempt_index,
        kind=attempt.kind.value,
        status=attempt.status.value,
        revision_of=attempt.revision_of,
        seed=attempt.seed,
        prompt=attempt.prompt,
        model_id=output.model_id if output is not None else None,
        generation=generation_record,
        normalization=_dataclass_dict(attempt.normalization),
        validation=_dataclass_dict(attempt.validation),
        critic=critic,
        revision_decision=_dataclass_dict(attempt.revision_decision),
        artifacts=dict(artifacts),
    )


def build_run_metadata(
    result: GenerationRunResult,
    attempt_artifacts: Mapping[str, Mapping[str, str]] | None = None,
    run_artifacts: Mapping[str, str] | None = None,
) -> RunMetadata:
    """Return the JSON-safe record of one run; artifact paths are optional."""
    attempt_artifacts = attempt_artifacts or {}
    outcome = result.outcome
    return RunMetadata(
        schema_version=SCHEMA_VERSION,
        run_id=result.run_id,
        status=outcome.status.value,
        requires_user_approval=outcome.requires_user_approval,
        approved=False,
        started_at=result.started_at,
        finished_at=result.finished_at,
        duration_s=(result.finished_at - result.started_at).total_seconds(),
        user_prompt=result.prompt.user_prompt,
        prompt_used=result.prompt.text,
        automatic_prompt=result.prompt.is_automatic,
        seed=result.request.seed,
        generator_model_id=result.generator_model_id,
        critic_model_id=result.critic_model_id,
        simulated_validation=result.simulated_validation,
        config=result.config.to_dict(),
        counters=asdict(result.counters),
        revision_counts=dict(result.revision_counts),
        ranking=[_dataclass_dict(item) or {} for item in outcome.ranked],
        selections=[_dataclass_dict(item) or {} for item in outcome.selections],
        outcome_reasons=list(outcome.reasons),
        attempts=[
            _attempt_metadata(attempt, attempt_artifacts.get(attempt.attempt_id, {}))
            for attempt in result.attempts
        ],
        artifacts=dict(run_artifacts or {}),
    )


def write_json(path: Path, payload: str) -> None:
    """Write UTF-8 JSON text with a final newline."""
    path.write_text(payload.rstrip("\n") + "\n", encoding="utf-8")


def _write_png(path: Path, image: np.ndarray) -> None:
    pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 and image.shape[2] == 3 else image
    if not cv2.imwrite(str(path), pixels):
        raise OSError(f"failed to write {path}")


def _write_attempt_artifacts(directory: Path, root: Path, attempt: AttemptRecord) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    images: dict[str, RGBImage | BinaryMask | None] = {
        "raw_output": attempt.generation.output.image if attempt.generation.output else None,
        "candidate_rgb": attempt.normalized.candidate_rgb if attempt.normalized else None,
        "candidate_new_ink_mask": attempt.normalized.candidate_new_ink_mask if attempt.normalized else None,
        "sanitized_new_ink_mask": attempt.sanitized_new_ink_mask,
        "preview_rgb": attempt.preview_rgb,
    }
    written = {}
    for name, image in images.items():
        if image is None:
            continue
        path = directory / f"{name}.png"
        _write_png(path, image)
        written[name] = path.relative_to(root).as_posix()
    return written


def write_run_record(result: GenerationRunResult, output_dir: Path) -> Path:
    """Write images, masks, and ``run.json`` under ``output_dir``; return the JSON path."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_png(root / "source_page.png", result.request.canonical_page)
    _write_png(root / "region_guidance.png", result.guidance_image)
    run_artifacts = {"source_page": "source_page.png", "region_guidance": "region_guidance.png"}
    attempt_artifacts = {
        attempt.attempt_id: _write_attempt_artifacts(
            root / "attempts" / attempt.attempt_id.replace("/", "_"), root, attempt
        )
        for attempt in result.attempts
    }
    metadata = build_run_metadata(result, attempt_artifacts, run_artifacts)
    path = root / "run.json"
    write_json(path, metadata.model_dump_json(indent=2))
    return path
