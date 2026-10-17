"""Workflow configuration and its validated external (JSON) representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from printgpt_alignment.geometry import A4CalibrationLayout, DEFAULT_LAYOUT

from .critic import CriticThresholds
from .extraction import ExtractionConfig
from .models import JSONValue
from .providers import RetryPolicy
from .ranking import RevisionPolicy


# Each candidate visits at most ~12 nodes (initial pass, decision, revision pass,
# decision); 60 leaves headroom while still stopping any accidental cycle quickly.
DEFAULT_RECURSION_LIMIT = 60


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    layout: A4CalibrationLayout = DEFAULT_LAYOUT
    candidate_count: int = 2
    revision: RevisionPolicy = field(default_factory=RevisionPolicy)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    critic: CriticThresholds = field(default_factory=CriticThresholds)
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __post_init__(self) -> None:
        if self.candidate_count not in (1, 2):
            raise ValueError("candidate_count must be 1 or 2")
        if self.recursion_limit < 10:
            raise ValueError("recursion_limit is too small for one candidate")

    @property
    def max_generation_requests(self) -> int:
        return self.candidate_count * (1 + self.revision.max_revisions_per_candidate)

    def to_dict(self) -> dict[str, JSONValue]:
        result = asdict(self)
        result["layout"] = {
            "layout_id": self.layout.layout_id,
            "dpi": self.layout.dpi,
            "image_shape_rows_columns": list(self.layout.image_shape),
        }
        return result


class GenerationSettings(BaseModel):
    """Optional configuration file; unspecified fields keep the defaults above."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_count: int = Field(2, ge=1, le=2)
    max_revisions_per_candidate: int = Field(1, ge=0, le=1)
    revise_after_geometry_rejection: bool = True
    max_attempts_per_call: int = Field(2, ge=1, le=5)
    max_total_provider_attempts: int = Field(16, ge=1, le=64)
    provider_timeout_s: float = Field(120.0, gt=0, le=600)
    min_critic_score: float = Field(6.0, ge=0, le=10)
    min_sub_score: float = Field(5.0, ge=0, le=10)
    source_redraw_tolerance_mm: float = Field(1.0, ge=0)
    max_registration_shift_mm: float = Field(0.5, ge=0)
    auto_align: bool = True
    recursion_limit: int = Field(DEFAULT_RECURSION_LIMIT, ge=10, le=500)

    @classmethod
    def from_json_file(cls, path: Path) -> GenerationSettings:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_config(self, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> GenerationConfig:
        return GenerationConfig(
            layout=layout,
            candidate_count=self.candidate_count,
            revision=RevisionPolicy(self.max_revisions_per_candidate, self.revise_after_geometry_rejection),
            retry=RetryPolicy(self.max_attempts_per_call, self.max_total_provider_attempts, self.provider_timeout_s),
            extraction=ExtractionConfig(
                source_redraw_tolerance_mm=self.source_redraw_tolerance_mm,
                max_registration_shift_mm=self.max_registration_shift_mm,
                auto_align=self.auto_align,
            ),
            critic=CriticThresholds(
                self.min_critic_score, self.min_sub_score, self.min_sub_score, self.min_sub_score
            ),
            recursion_limit=self.recursion_limit,
        )
