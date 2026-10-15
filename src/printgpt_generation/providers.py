"""Provider interfaces, bounded transport retries, and injectable clock/ID sources.

Adapters raise the ``ProviderError`` subclasses below for expected failures;
anything else is a programming error and propagates. Transport retries happen
here and never count as semantic revisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

from .models import (
    AttemptKind,
    BinaryMask,
    GeneratedImage,
    ProviderFailure,
    ProviderFailureKind,
    RGBImage,
    UsageRecord,
)


T = TypeVar("T")


class ProviderError(Exception):
    """Expected provider failure that must become a typed result state."""


class ProviderTransientError(ProviderError):
    """Failure that may succeed when retried (rate limit, connection reset)."""


class ProviderTimeoutError(ProviderTransientError):
    """The adapter's own deadline elapsed before the provider answered."""


class ProviderPermanentError(ProviderError):
    """Failure that retrying cannot fix (refusal, invalid request, no image returned)."""


@dataclass(frozen=True, slots=True, eq=False)
class GenerationCall:
    """Everything a generator needs for one candidate attempt.

    ``printable_region_mask`` lets adapters render their own guidance; it is
    never sent as printable content. ``previous_candidate`` is set for revisions.
    """

    candidate_id: str
    candidate_index: int
    attempt_index: int
    kind: AttemptKind
    prompt: str
    seed: int
    source_page: RGBImage
    guidance_image: RGBImage
    printable_region_mask: BinaryMask
    previous_candidate: RGBImage | None = None


@dataclass(frozen=True, slots=True, eq=False)
class CritiqueCall:
    """Critic inputs. The preview is the sanitized composite, never raw model output."""

    candidate_id: str
    candidate_index: int
    attempt_index: int
    prompt: str
    original_page: RGBImage
    guidance_image: RGBImage
    sanitized_preview: RGBImage


@dataclass(frozen=True, slots=True)
class RawCritique:
    """Untrusted critic text; ``critic.parse_critic_output`` validates it."""

    text: str
    model_id: str
    usage: UsageRecord = UsageRecord()


class CandidateGenerator(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(self, call: GenerationCall) -> GeneratedImage: ...


class VisualCritic(Protocol):
    @property
    def model_id(self) -> str: ...

    def critique(self, call: CritiqueCall) -> RawCritique: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdProvider(Protocol):
    def new_id(self, prefix: str) -> str: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidIdProvider:
    def new_id(self, prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:12]}"


class FixedStepClock:
    """Deterministic clock advancing by a fixed step on every reading."""

    def __init__(self, start: datetime | None = None, step: timedelta = timedelta(milliseconds=10)) -> None:
        if step < timedelta(0):
            raise ValueError("clock step cannot be negative")
        self._current = start or datetime(2026, 1, 1, tzinfo=UTC)
        self._step = step

    def now(self) -> datetime:
        current = self._current
        self._current = current + self._step
        return current


class SequentialIdProvider:
    """Deterministic IDs such as ``candidate-1``; counters are per prefix."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def new_id(self, prefix: str) -> str:
        value = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = value
        return f"{prefix}-{value}"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Transport retry bounds.

    ``max_attempts_per_call`` bounds one logical call; ``max_total_attempts``
    bounds a whole run so retries can never create an unbounded loop. Deadlines
    are enforced by adapters (see ``timeout_s``) because synchronous calls cannot
    be pre-empted safely from here.
    """

    max_attempts_per_call: int = 2
    max_total_attempts: int = 16
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if self.max_attempts_per_call < 1:
            raise ValueError("max_attempts_per_call must be at least 1")
        if self.max_total_attempts < 1:
            raise ValueError("max_total_attempts must be at least 1")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclass(frozen=True, slots=True)
class ProviderCallResult(Generic[T]):
    value: T | None
    failure: ProviderFailure | None
    attempts: int
    latency_s: float


def call_with_retries(
    operation: Callable[[], T],
    *,
    max_attempts: int,
    clock: Clock,
) -> ProviderCallResult[T]:
    """Run ``operation`` at most ``max_attempts`` times, retrying transient failures only."""
    started = clock.now()
    if max_attempts < 1:
        failure = ProviderFailure(ProviderFailureKind.BUDGET_EXHAUSTED, "provider attempt budget exhausted")
        return ProviderCallResult(None, failure, 0, 0.0)
    attempts = 0
    failure: ProviderFailure | None = None
    while attempts < max_attempts:
        attempts += 1
        try:
            value = operation()
        except ProviderTimeoutError as error:
            failure = ProviderFailure(ProviderFailureKind.TIMEOUT, str(error) or "provider timed out")
        except ProviderTransientError as error:
            failure = ProviderFailure(ProviderFailureKind.ERROR, str(error) or "transient provider error")
        except ProviderPermanentError as error:
            failure = ProviderFailure(ProviderFailureKind.ERROR, str(error) or "provider error")
            break
        else:
            return ProviderCallResult(value, None, attempts, _elapsed_s(started, clock))
    return ProviderCallResult(None, failure, attempts, _elapsed_s(started, clock))


def _elapsed_s(started: datetime, clock: Clock) -> float:
    return max(0.0, (clock.now() - started).total_seconds())

