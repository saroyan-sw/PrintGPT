"""Typed domain values for continuation generation and candidate review.

Images and masks follow the shared contract: RGB ``uint8`` pages and binary
``uint8`` masks in canonical (rows, columns) order, where 255 means membership
in the named mask. Arrays held by these records are read-only copies so graph
nodes can share them without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


RGBImage: TypeAlias = NDArray[np.uint8]
BinaryMask: TypeAlias = NDArray[np.uint8]
JSONValue: TypeAlias = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def require_rgb_image(image: object, expected_shape: tuple[int, int], name: str) -> None:
    """Raise ``ValueError`` unless ``image`` is an RGB uint8 page of the expected size."""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    required_shape = (*expected_shape, 3)
    if image.shape != required_shape:
        raise ValueError(f"{name} must have shape {required_shape}, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {image.dtype}")


def require_binary_mask(mask: object, expected_shape: tuple[int, int], name: str) -> None:
    """Raise ``ValueError`` unless ``mask`` is a 0/255 uint8 mask of the expected size."""
    if mask is None:
        raise ValueError(f"{name} is required")
    if not isinstance(mask, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {mask.shape}")
    if mask.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {mask.dtype}")
    if not bool(np.all((mask == 0) | (mask == 255))):
        raise ValueError(f"{name} values must be exactly 0 or 255")


def readonly_copy(array: np.ndarray) -> np.ndarray:
    """Return a contiguous copy that cannot be modified in place."""
    result = np.ascontiguousarray(array.copy())
    result.flags.writeable = False
    return result


class AttemptKind(StrEnum):
    INITIAL = "initial"
    REVISION = "revision"


class AttemptStatus(StrEnum):
    """Furthest stage one generation attempt reached; graph routing keys off this."""

    GENERATED = "generated"
    GENERATION_FAILED = "generation_failed"
    GEOMETRY_REJECTED = "geometry_rejected"
    EXTRACTED = "extracted"
    VALIDATION_REJECTED = "validation_rejected"
    VALIDATED = "validated"
    CRITIC_FAILED = "critic_failed"
    CRITICIZED = "criticized"


class ProviderFailureKind(StrEnum):
    ERROR = "error"
    TIMEOUT = "timeout"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MALFORMED_OUTPUT = "malformed_output"


class CriticAssessment(StrEnum):
    """Deterministic interpretation of a schema-valid critic report."""

    ACCEPTABLE = "acceptable"
    NEEDS_REVISION = "needs_revision"
    UNACCEPTABLE = "unacceptable"


class RevisionDecisionKind(StrEnum):
    REVISE = "revise"
    KEEP = "keep"
    NOT_REVISABLE = "not_revisable"
    LIMIT_REACHED = "limit_reached"


class RunStatus(StrEnum):
    CANDIDATES_READY = "candidates_ready"
    NO_VALID_CANDIDATE = "no_valid_candidate"


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """Provider-reported usage; ``None`` means unknown rather than zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    kind: ProviderFailureKind
    message: str


@dataclass(frozen=True, slots=True, eq=False)
class GeneratedImage:
    """Raw provider output, kept exactly as returned apart from being read-only.

    ``image`` may have any size or channel layout; normalization decides whether
    it is supported. ``additions_mask`` is an optional explicit additions layer
    in the same pixel grid as ``image``. ``canvas_box`` is ``(x, y, width,
    height)`` in canonical pixels when ``image`` depicts only that window of the
    page (crop framing); ``None`` means the complete page.
    """

    image: np.ndarray
    model_id: str
    additions_mask: np.ndarray | None = None
    usage: UsageRecord = field(default_factory=UsageRecord)
    canvas_box: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray) or self.image.size == 0:
            raise ValueError("generated image must be a non-empty NumPy array")
        if self.canvas_box is not None:
            box = tuple(int(value) for value in self.canvas_box)
            if len(box) != 4 or box[0] < 0 or box[1] < 0 or box[2] < 1 or box[3] < 1:
                raise ValueError(f"canvas_box must be (x, y, width, height) with a positive size, got {self.canvas_box}")
            object.__setattr__(self, "canvas_box", box)
        object.__setattr__(self, "image", readonly_copy(self.image))
        if self.additions_mask is not None:
            if self.additions_mask.shape[:2] != self.image.shape[:2]:
                raise ValueError("additions_mask must share the generated image's pixel grid")
            object.__setattr__(self, "additions_mask", readonly_copy(self.additions_mask))


@dataclass(frozen=True, slots=True, eq=False)
class GenerationOutcome:
    output: GeneratedImage | None
    failure: ProviderFailure | None
    provider_attempts: int
    latency_s: float


@dataclass(frozen=True, slots=True)
class RegistrationReport:
    """Evidence that the source page survived generation in place."""

    verified: bool
    estimated_shift_px: tuple[float, float]
    estimated_shift_mm: float
    phase_response: float
    source_ink_pixels: int
    source_retention: float


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    method: str
    candidate_ink_pixels: int
    discarded_source_redraw_pixels: int
    discarded_noise_pixels: int
    discarded_coloured_pixels: int
    new_ink_pixels: int


@dataclass(frozen=True, slots=True)
class NormalizationRecord:
    """Recorded mapping from raw provider pixels to canonical page pixels."""

    input_shape: tuple[int, ...]
    raw_to_canonical: tuple[tuple[float, float, float], ...]
    aspect_ratio_error: float
    registration: RegistrationReport | None
    extraction: ExtractionReport | None
    failure_reasons: tuple[str, ...] = ()
    # Window of the page the raw output depicts; registration was measured inside it.
    canvas_box: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True, eq=False)
class NormalizedCandidate:
    candidate_rgb: RGBImage
    candidate_new_ink_mask: BinaryMask


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """JSON-safe copy of a Track A ``ValidationResult`` without its raster."""

    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict[str, JSONValue]
    simulated: bool


@dataclass(frozen=True, slots=True)
class CriticScores:
    score: float
    boundary_continuity_score: float
    style_match_score: float
    geometry_plausibility_score: float


@dataclass(frozen=True, slots=True)
class CriticVerdict:
    scores: CriticScores
    problems: tuple[str, ...]
    revision_instruction: str
    recommendation: str
    assessment: CriticAssessment


@dataclass(frozen=True, slots=True)
class CriticOutcome:
    verdict: CriticVerdict | None
    failure: ProviderFailure | None
    provider_attempts: int
    latency_s: float
    model_id: str
    usage: UsageRecord = field(default_factory=UsageRecord)


@dataclass(frozen=True, slots=True)
class RevisionDecision:
    kind: RevisionDecisionKind
    reasons: tuple[str, ...]
    instruction: str = ""


@dataclass(frozen=True, slots=True, eq=False)
class AttemptRecord:
    """Everything known about one generation attempt of one candidate.

    Nodes replace records with ``dataclasses.replace`` instead of mutating
    them, so earlier snapshots remain valid in metadata and tests.
    """

    attempt_id: str
    candidate_id: str
    candidate_index: int
    attempt_index: int
    kind: AttemptKind
    status: AttemptStatus
    prompt: str
    seed: int
    generation: GenerationOutcome
    revision_of: str | None = None
    normalization: NormalizationRecord | None = None
    normalized: NormalizedCandidate | None = None
    validation: ValidationSummary | None = None
    sanitized_new_ink_mask: BinaryMask | None = None
    preview_rgb: RGBImage | None = None
    critic: CriticOutcome | None = None
    revision_decision: RevisionDecision | None = None


@dataclass(frozen=True, slots=True)
class CallCounters:
    """Logical requests and transport attempts, counted separately from revisions."""

    generation_requests: int = 0
    critic_requests: int = 0
    provider_attempts: int = 0


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    rank: int
    candidate_id: str
    attempt_id: str
    score: float
    used_revision: bool


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """The attempt kept for one candidate, or ``None`` with an explanation."""

    candidate_id: str
    selected_attempt_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """Terminal result. Selection is a proposal only: it is never approved or printed."""

    status: RunStatus
    ranked: tuple[RankedCandidate, ...]
    selections: tuple[CandidateSelection, ...]
    reasons: tuple[str, ...]
    requires_user_approval: bool = True
