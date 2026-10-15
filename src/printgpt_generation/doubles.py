"""Deterministic generator and critic doubles for offline runs and tests.

Behaviour is scripted per ``(candidate_index, attempt_index)`` and all
randomness comes from the call's seed, so identical inputs give identical
pixels. The doubles record their calls for assertions; that state is per
instance, never module-level.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import json
from math import cos, pi, sin

import cv2
import numpy as np

from .guidance import OUTLINE_RGB
from .models import GeneratedImage
from .providers import (
    CritiqueCall,
    GenerationCall,
    ProviderPermanentError,
    ProviderTimeoutError,
    ProviderTransientError,
    RawCritique,
)


AttemptKey = tuple[int, int]
"""``(candidate_index, attempt_index)``; attempt 0 is initial, 1 is the revision."""


class GeneratorBehavior(StrEnum):
    CONTINUE = "continue"
    BLANK = "blank"
    OUTSIDE_REGION = "outside_region"
    SHIFT_SOURCE = "shift_source"
    REDRAW_SOURCE = "redraw_source"
    NOISY = "noisy"
    GUIDANCE_ECHO = "guidance_echo"
    WRONG_ASPECT = "wrong_aspect"
    HALF_SCALE = "half_scale"
    ADDITIONS_LAYER = "additions_layer"
    ERROR = "error"
    TIMEOUT = "timeout"


def _stroke_thickness(image: np.ndarray) -> int:
    return max(2, round(image.shape[1] / 400))


def _draw_stroke_inside(
    image: np.ndarray, region_mask: np.ndarray, seed: int, colour: int | tuple[int, int, int] = (0, 0, 0)
) -> None:
    """Draw one seeded straight stroke that stays inside ``region_mask``."""
    region = (region_mask > 0).astype(np.uint8)
    thickness = _stroke_thickness(region_mask)
    distances = cv2.distanceTransform(region, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_distance = float(distances.max())
    if max_distance < 3 * thickness:
        return
    rng = np.random.default_rng(seed)
    centres = np.argwhere(distances >= 0.6 * max_distance)
    y, x = centres[int(rng.integers(len(centres)))]
    # A segment shorter than the clearance radius around its midpoint stays in the region.
    half_length = 0.7 * float(distances[y, x]) - thickness
    angle = float(rng.uniform(0.0, pi))
    dx, dy = half_length * cos(angle), half_length * sin(angle)
    start = (int(round(x - dx)), int(round(y - dy)))
    end = (int(round(x + dx)), int(round(y + dy)))
    cv2.line(image, start, end, colour, thickness)


def _draw_outside_stroke(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    y = max(1, height // 60)
    cv2.line(image, (width // 10, y), (9 * width // 10, y), (0, 0, 0), _stroke_thickness(image))


def _source_ink(page: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(page, cv2.COLOR_RGB2GRAY) < 128


class DeterministicGenerator:
    """Generator double whose output is a pure function of the call and script."""

    def __init__(
        self,
        behaviors: Mapping[AttemptKey, GeneratorBehavior] | None = None,
        *,
        default: GeneratorBehavior = GeneratorBehavior.CONTINUE,
        transient_failures: Mapping[AttemptKey, int] | None = None,
        shift_px: int = 40,
        model_id: str = "deterministic-generator-v1",
    ) -> None:
        self._behaviors = dict(behaviors or {})
        self._default = default
        self._transient_failures = dict(transient_failures or {})
        self._shift_px = shift_px
        self._model_id = model_id
        self.calls: list[GenerationCall] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def behavior_for(self, call: GenerationCall) -> GeneratorBehavior:
        return self._behaviors.get((call.candidate_index, call.attempt_index), self._default)

    def generate(self, call: GenerationCall) -> GeneratedImage:
        self.calls.append(call)
        key = (call.candidate_index, call.attempt_index)
        previous_failures = sum(
            1 for earlier in self.calls[:-1] if (earlier.candidate_index, earlier.attempt_index) == key
        )
        if previous_failures < self._transient_failures.get(key, 0):
            raise ProviderTransientError("simulated rate limit")
        behavior = self.behavior_for(call)
        if behavior is GeneratorBehavior.ERROR:
            raise ProviderPermanentError("simulated provider refusal")
        if behavior is GeneratorBehavior.TIMEOUT:
            raise ProviderTimeoutError("simulated timeout")
        return self._render(call, behavior)

    def _render(self, call: GenerationCall, behavior: GeneratorBehavior) -> GeneratedImage:
        page = np.array(call.source_page, copy=True)
        additions = np.zeros(page.shape[:2], dtype=np.uint8)
        if behavior is GeneratorBehavior.SHIFT_SOURCE:
            matrix = np.float32([[1, 0, self._shift_px], [0, 1, self._shift_px // 2]])
            page = cv2.warpAffine(page, matrix, (page.shape[1], page.shape[0]), borderValue=(255, 255, 255))
        if behavior is GeneratorBehavior.REDRAW_SOURCE:
            # Models often re-ink originals slightly thicker and offset by a pixel.
            redrawn = cv2.dilate(_source_ink(page).astype(np.uint8), np.ones((2, 2), np.uint8))
            redrawn = np.roll(redrawn, 1, axis=1) > 0
            page[redrawn] = (0, 0, 0)

        if behavior is GeneratorBehavior.OUTSIDE_REGION:
            _draw_outside_stroke(page)
        elif behavior is not GeneratorBehavior.BLANK:
            _draw_stroke_inside(page, call.printable_region_mask, call.seed)
            _draw_stroke_inside(additions, call.printable_region_mask, call.seed, 255)

        if behavior is GeneratorBehavior.NOISY:
            rng = np.random.default_rng(call.seed + 7919)
            speckles = rng.random(page.shape[:2]) < 0.002
            page[speckles] = (0, 0, 0)
            light_noise = rng.integers(0, 25, size=page.shape, dtype=np.int16)
            page = np.clip(page.astype(np.int16) - light_noise, 0, 255).astype(np.uint8)
        if behavior is GeneratorBehavior.GUIDANCE_ECHO:
            outline = np.all(call.guidance_image == np.asarray(OUTLINE_RGB, dtype=np.uint8), axis=2)
            page[outline] = OUTLINE_RGB
        if behavior is GeneratorBehavior.WRONG_ASPECT:
            side = min(page.shape[:2])
            page = cv2.resize(page, (side, side), interpolation=cv2.INTER_AREA)
        if behavior is GeneratorBehavior.HALF_SCALE:
            page = cv2.resize(page, (page.shape[1] // 2, page.shape[0] // 2), interpolation=cv2.INTER_AREA)

        layer = additions if behavior is GeneratorBehavior.ADDITIONS_LAYER else None
        return GeneratedImage(image=page, model_id=self._model_id, additions_mask=layer)


class CriticBehavior(StrEnum):
    RESPOND = "respond"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VIOLATION = "schema_violation"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class CriticScript:
    behavior: CriticBehavior = CriticBehavior.RESPOND
    recommendation: str = "accept"
    score: float = 8.0
    sub_score: float = 7.5
    revision_instruction: str = ""
    problems: tuple[str, ...] = ()

    @classmethod
    def accept(cls, score: float = 8.0) -> CriticScript:
        return cls(score=score)

    @classmethod
    def revise(cls, score: float = 4.0, instruction: str = "Extend the stroke along the existing contour.") -> CriticScript:
        return cls(
            recommendation="revise",
            score=score,
            sub_score=score,
            revision_instruction=instruction,
            problems=("continuation does not follow the existing contour",),
        )

    @classmethod
    def reject(cls) -> CriticScript:
        return cls(recommendation="reject", score=1.0, sub_score=1.0, problems=("unrelated object",))


class DeterministicCritic:
    """Critic double that returns scripted JSON text and records what it was shown."""

    def __init__(
        self,
        scripts: Mapping[AttemptKey, CriticScript] | None = None,
        *,
        default: CriticScript = CriticScript(),
        model_id: str = "deterministic-critic-v1",
    ) -> None:
        self._scripts = dict(scripts or {})
        self._default = default
        self._model_id = model_id
        self.calls: list[CritiqueCall] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def critique(self, call: CritiqueCall) -> RawCritique:
        self.calls.append(call)
        script = self._scripts.get((call.candidate_index, call.attempt_index), self._default)
        if script.behavior is CriticBehavior.ERROR:
            raise ProviderPermanentError("simulated critic failure")
        if script.behavior is CriticBehavior.TIMEOUT:
            raise ProviderTimeoutError("simulated critic timeout")
        if script.behavior is CriticBehavior.MALFORMED_JSON:
            return RawCritique("The drawing looks fine overall, score 8.", self._model_id)
        payload = {
            "score": script.score,
            "boundary_continuity_score": script.sub_score,
            "style_match_score": script.sub_score,
            "geometry_plausibility_score": script.sub_score,
            "problems": list(script.problems),
            "revision_instruction": script.revision_instruction,
            "recommendation": script.recommendation,
        }
        if script.behavior is CriticBehavior.SCHEMA_VIOLATION:
            payload["score"] = 15
            payload["recommendation"] = "maybe"
        return RawCritique(json.dumps(payload), self._model_id)
