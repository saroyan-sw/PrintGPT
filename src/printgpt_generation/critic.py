"""Critic response schema, parsing, and deterministic interpretation.

Critic scores are heuristic judgments, not calibrated probabilities. The
critic can only ask for a better continuation; it never sees or changes the
masks, and its verdict cannot make a validation-rejected candidate eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .models import CriticAssessment, CriticScores, CriticVerdict


Score = Annotated[float, Field(ge=0.0, le=10.0)]


class CriticReport(BaseModel):
    """Schema for untrusted critic JSON."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: Score
    boundary_continuity_score: Score
    style_match_score: Score
    geometry_plausibility_score: Score
    problems: list[Annotated[str, Field(max_length=500)]] = Field(max_length=20)
    revision_instruction: Annotated[str, Field(max_length=2000)]
    recommendation: Literal["accept", "revise", "reject"]


@dataclass(frozen=True, slots=True)
class CriticThresholds:
    min_score: float = 6.0
    min_boundary_continuity_score: float = 5.0
    min_style_match_score: float = 5.0
    min_geometry_plausibility_score: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "min_score",
            "min_boundary_continuity_score",
            "min_style_match_score",
            "min_geometry_plausibility_score",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 10.0:
                raise ValueError(f"{name} must be in [0, 10]")


@dataclass(frozen=True, slots=True)
class CriticParseResult:
    report: CriticReport | None
    error: str | None


_FENCE = re.compile(r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)


def parse_critic_output(text: str) -> CriticParseResult:
    """Validate critic text against ``CriticReport``; malformed output is a typed error."""
    match = _FENCE.match(text)
    body = match.group("body") if match else text
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        return CriticParseResult(None, f"critic output is not valid JSON: {error.msg}")
    try:
        return CriticParseResult(CriticReport.model_validate(payload), None)
    except ValidationError as error:
        summary = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc']) or 'root'}: {issue['msg']}"
            for issue in error.errors()[:5]
        )
        return CriticParseResult(None, f"critic output failed schema validation: {summary}")


def meets_thresholds(scores: CriticScores, thresholds: CriticThresholds) -> bool:
    return (
        scores.score >= thresholds.min_score
        and scores.boundary_continuity_score >= thresholds.min_boundary_continuity_score
        and scores.style_match_score >= thresholds.min_style_match_score
        and scores.geometry_plausibility_score >= thresholds.min_geometry_plausibility_score
    )


def assess(recommendation: str, scores: CriticScores, thresholds: CriticThresholds) -> CriticAssessment:
    """Combine the critic's recommendation with configured score thresholds."""
    if recommendation == "reject":
        return CriticAssessment.UNACCEPTABLE
    if recommendation == "revise" or not meets_thresholds(scores, thresholds):
        return CriticAssessment.NEEDS_REVISION
    return CriticAssessment.ACCEPTABLE


def to_verdict(report: CriticReport, thresholds: CriticThresholds) -> CriticVerdict:
    scores = CriticScores(
        score=report.score,
        boundary_continuity_score=report.boundary_continuity_score,
        style_match_score=report.style_match_score,
        geometry_plausibility_score=report.geometry_plausibility_score,
    )
    return CriticVerdict(
        scores=scores,
        problems=tuple(report.problems),
        revision_instruction=report.revision_instruction,
        recommendation=report.recommendation,
        assessment=assess(report.recommendation, scores, thresholds),
    )


def is_rankable(verdict: CriticVerdict, thresholds: CriticThresholds) -> bool:
    """A verdict may be ranked when it is not a rejection and meets every threshold.

    A ``revise`` recommendation that still meets thresholds remains rankable so
    the original can be kept if its revision fails or scores lower.
    """
    return verdict.recommendation != "reject" and meets_thresholds(verdict.scores, thresholds)
