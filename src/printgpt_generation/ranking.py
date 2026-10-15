"""Deterministic revision, eligibility, and ranking rules.

Eligibility requires accepted deterministic validation AND a completed,
rankable critic verdict. Nothing here can promote a validation-rejected
attempt, and the ranking is a proposal for human review, never an approval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .critic import CriticThresholds, is_rankable
from .models import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    CandidateSelection,
    CriticAssessment,
    RankedCandidate,
    RevisionDecision,
    RevisionDecisionKind,
    RunOutcome,
    RunStatus,
)


@dataclass(frozen=True, slots=True)
class RevisionPolicy:
    max_revisions_per_candidate: int = 1
    revise_after_geometry_rejection: bool = True

    def __post_init__(self) -> None:
        if self.max_revisions_per_candidate not in (0, 1):
            raise ValueError("max_revisions_per_candidate must be 0 or 1")


def is_eligible(attempt: AttemptRecord, thresholds: CriticThresholds) -> bool:
    return (
        attempt.status is AttemptStatus.CRITICIZED
        and attempt.validation is not None
        and attempt.validation.accepted
        and attempt.critic is not None
        and attempt.critic.verdict is not None
        and is_rankable(attempt.critic.verdict, thresholds)
    )


def decide_revision(attempt: AttemptRecord, revisions_used: int, policy: RevisionPolicy) -> RevisionDecision:
    """Decide whether this attempt's candidate gets its single revision."""
    if attempt.kind is AttemptKind.REVISION or revisions_used >= policy.max_revisions_per_candidate:
        return RevisionDecision(RevisionDecisionKind.LIMIT_REACHED, ("revision limit reached",))
    status = attempt.status
    if status is AttemptStatus.VALIDATION_REJECTED and attempt.validation is not None:
        return RevisionDecision(RevisionDecisionKind.REVISE, attempt.validation.reasons)
    if status is AttemptStatus.GEOMETRY_REJECTED and policy.revise_after_geometry_rejection:
        reasons = attempt.normalization.failure_reasons if attempt.normalization else ()
        return RevisionDecision(
            RevisionDecisionKind.REVISE,
            reasons,
            "Return the full page at the original framing with existing marks unmoved.",
        )
    if status is AttemptStatus.CRITICIZED and attempt.critic is not None and attempt.critic.verdict is not None:
        verdict = attempt.critic.verdict
        if verdict.assessment is CriticAssessment.ACCEPTABLE:
            return RevisionDecision(RevisionDecisionKind.KEEP, ("critic accepted the candidate",))
        return RevisionDecision(RevisionDecisionKind.REVISE, verdict.problems, verdict.revision_instruction)
    # Transport and critic failures carry no semantic feedback to revise against.
    return RevisionDecision(RevisionDecisionKind.NOT_REVISABLE, (f"attempt ended as {status.value}",))


def _score(attempt: AttemptRecord) -> float:
    assert attempt.critic is not None and attempt.critic.verdict is not None
    return attempt.critic.verdict.scores.score


def _sub_score_mean(attempt: AttemptRecord) -> float:
    assert attempt.critic is not None and attempt.critic.verdict is not None
    scores = attempt.critic.verdict.scores
    return (
        scores.boundary_continuity_score + scores.style_match_score + scores.geometry_plausibility_score
    ) / 3.0


def select_attempt(
    candidate_id: str,
    attempts: Sequence[AttemptRecord],
    thresholds: CriticThresholds,
) -> CandidateSelection:
    """Keep the revision only when it is eligible and not worse than an eligible original."""
    ordered = sorted(attempts, key=lambda attempt: attempt.attempt_index)
    original = ordered[0] if ordered else None
    revision = ordered[1] if len(ordered) > 1 else None
    original_ok = original is not None and is_eligible(original, thresholds)
    revision_ok = revision is not None and is_eligible(revision, thresholds)
    if revision_ok and (not original_ok or _score(revision) >= _score(original)):
        return CandidateSelection(candidate_id, revision.attempt_id, "revision is eligible and not worse")
    if original_ok:
        reason = "original kept; revision ineligible or scored lower" if revision else "original is eligible"
        return CandidateSelection(candidate_id, original.attempt_id, reason)
    last = ordered[-1].status.value if ordered else "no attempts"
    return CandidateSelection(candidate_id, None, f"no eligible attempt (last status: {last})")


def build_outcome(
    candidate_ids: Sequence[str],
    attempts: Sequence[AttemptRecord],
    thresholds: CriticThresholds,
) -> RunOutcome:
    """Select one attempt per candidate and rank eligible selections deterministically."""
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    selections = tuple(
        select_attempt(
            candidate_id,
            [attempt for attempt in attempts if attempt.candidate_id == candidate_id],
            thresholds,
        )
        for candidate_id in candidate_ids
    )
    chosen = [by_id[selection.selected_attempt_id] for selection in selections if selection.selected_attempt_id]
    # Ties fall back to sub-scores, then less clipping, then candidate order, so reruns agree.
    chosen.sort(
        key=lambda attempt: (
            -_score(attempt),
            -_sub_score_mean(attempt),
            float(attempt.validation.metrics.get("removed_fraction", 0.0) or 0.0) if attempt.validation else 0.0,
            attempt.candidate_index,
        )
    )
    ranked = tuple(
        RankedCandidate(
            rank=position + 1,
            candidate_id=attempt.candidate_id,
            attempt_id=attempt.attempt_id,
            score=_score(attempt),
            used_revision=attempt.kind is AttemptKind.REVISION,
        )
        for position, attempt in enumerate(chosen)
    )
    if not ranked:
        reasons = tuple(f"{selection.candidate_id}: {selection.reason}" for selection in selections)
        return RunOutcome(RunStatus.NO_VALID_CANDIDATE, (), selections, reasons)
    return RunOutcome(RunStatus.CANDIDATES_READY, ranked, selections, ())
