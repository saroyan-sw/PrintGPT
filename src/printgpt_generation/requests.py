"""Workflow input and result records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from .config import GenerationConfig
from .models import AttemptRecord, CallCounters, RGBImage, RunOutcome, readonly_copy, require_rgb_image
from .prompts import ResolvedPrompt
from .track_a import MaskBundleLike, require_mask_bundle


@dataclass(frozen=True, slots=True, eq=False)
class GenerationRequest:
    """Immutable workflow input; build it with ``create_generation_request``."""

    canonical_page: RGBImage
    masks: MaskBundleLike
    user_prompt: str | None
    seed: int


def create_generation_request(
    canonical_page: RGBImage,
    masks: MaskBundleLike,
    user_prompt: str | None = None,
    *,
    config: GenerationConfig,
    seed: int = 0,
) -> GenerationRequest:
    """Validate the canonical RGB page and Track A masks against the configured layout."""
    require_rgb_image(canonical_page, config.layout.image_shape, "canonical_page")
    require_mask_bundle(masks, config.layout.image_shape)
    if user_prompt is not None and not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a string or None")
    if seed < 0:
        raise ValueError("seed cannot be negative")
    return GenerationRequest(readonly_copy(canonical_page), masks, user_prompt, int(seed))


@dataclass(frozen=True, slots=True, eq=False)
class GenerationRunResult:
    """Final workflow record. Candidates are proposals awaiting human approval."""

    run_id: str
    request: GenerationRequest
    prompt: ResolvedPrompt
    guidance_image: RGBImage
    candidate_ids: tuple[str, ...]
    attempts: tuple[AttemptRecord, ...]
    revision_counts: Mapping[str, int]
    counters: CallCounters
    outcome: RunOutcome
    started_at: datetime
    finished_at: datetime
    generator_model_id: str
    critic_model_id: str
    simulated_validation: bool
    config: GenerationConfig

    def attempt(self, attempt_id: str) -> AttemptRecord:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise KeyError(attempt_id)
