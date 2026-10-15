import json

import pytest

from printgpt_generation.critic import CriticThresholds, parse_critic_output, to_verdict
from printgpt_generation.models import CriticAssessment
from printgpt_generation.prompts import (
    DEFAULT_CONTINUATION_PROMPT,
    build_generation_prompt,
    build_revision_prompt,
    resolve_user_prompt,
)


def _critic_json(**overrides: object) -> str:
    payload = {
        "score": 8,
        "boundary_continuity_score": 7,
        "style_match_score": 7,
        "geometry_plausibility_score": 7,
        "problems": [],
        "revision_instruction": "",
        "recommendation": "accept",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.mark.parametrize("user_prompt", [None, "", "   \n\t"])
def test_empty_prompt_uses_automatic_continuation(user_prompt: str | None) -> None:
    resolved = resolve_user_prompt(user_prompt)
    assert resolved.is_automatic
    assert resolved.text == DEFAULT_CONTINUATION_PROMPT
    assert build_generation_prompt(resolved).startswith(DEFAULT_CONTINUATION_PROMPT)


def test_default_prompt_matches_specification() -> None:
    assert DEFAULT_CONTINUATION_PROMPT == (
        "Continue the unfinished drawing naturally. Match its line thickness, style, perspective, "
        "and subject. Keep existing content unchanged. Add content only inside the supplied "
        "printable region. Do not add text, borders, backgrounds, shading, or unrelated objects. "
        "Respect excluded areas even if this leaves a small gap at an existing stroke endpoint."
    )


def test_user_prompt_is_kept_with_safety_rules() -> None:
    resolved = resolve_user_prompt("  add a tail to the cat ")
    assert not resolved.is_automatic
    prompt = build_generation_prompt(resolved)
    assert "add a tail to the cat" in prompt
    assert DEFAULT_CONTINUATION_PROMPT in prompt


def test_revision_prompt_carries_reasons_and_instruction() -> None:
    prompt = build_revision_prompt(resolve_user_prompt(""), ("removed fraction too high",), "Stay inside green.")
    assert "removed fraction too high" in prompt
    assert "Stay inside green." in prompt


def test_non_string_prompt_is_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_user_prompt(42)  # type: ignore[arg-type]


def test_critic_output_parses_plain_and_fenced_json() -> None:
    assert parse_critic_output(_critic_json()).report is not None
    assert parse_critic_output(f"```json\n{_critic_json()}\n```").report is not None


@pytest.mark.parametrize(
    "text",
    [
        "looks good, 8/10",
        _critic_json(score=11),
        _critic_json(recommendation="maybe"),
        _critic_json(extra_field=True),
        json.dumps({"score": 8}),
    ],
)
def test_malformed_critic_output_is_a_typed_error(text: str) -> None:
    parsed = parse_critic_output(text)
    assert parsed.report is None
    assert parsed.error


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, CriticAssessment.ACCEPTABLE),
        ({"recommendation": "revise"}, CriticAssessment.NEEDS_REVISION),
        ({"score": 3}, CriticAssessment.NEEDS_REVISION),
        ({"style_match_score": 2}, CriticAssessment.NEEDS_REVISION),
        ({"recommendation": "reject"}, CriticAssessment.UNACCEPTABLE),
    ],
)
def test_critic_assessment_combines_recommendation_and_thresholds(
    overrides: dict[str, object], expected: CriticAssessment
) -> None:
    report = parse_critic_output(_critic_json(**overrides)).report
    assert report is not None
    assert to_verdict(report, CriticThresholds()).assessment is expected
