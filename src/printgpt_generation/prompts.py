"""Prompt construction for generation, revision, and visual criticism."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Final

from .models import JSONValue


DEFAULT_CONTINUATION_PROMPT: Final[str] = (
    "Continue the unfinished drawing naturally. Match its line thickness, style, "
    "perspective, and subject. Keep existing content unchanged. Add content only "
    "inside the supplied printable region. Do not add text, borders, backgrounds, "
    "shading, or unrelated objects. Respect excluded areas even if this leaves a "
    "small gap at an existing stroke endpoint."
)

# The guidance image is a separate input; telling the model what its colours mean
# keeps them from being mistaken for drawing content it should reproduce.
_GUIDANCE_DESCRIPTION: Final[str] = (
    "You receive two images. The first is the page area to work on: the original "
    "page or a crop of it. The second is region guidance for exactly the same area: "
    "the printable region is tinted green and outlined in magenta; forbidden areas "
    "are tinted red. Never draw the tints, outline, or any guidance mark. Return "
    "the same area as the first image, at the same size, orientation, and framing, "
    "with every existing mark in exactly the same place, using black lines on the "
    "existing white paper."
)

# Per-candidate directions; indices past the end reuse the last one. Manual tests
# (October 2026) showed that asking any candidate for a "different" continuation
# made Gemini invent a new scene or redraw the whole page, so every candidate
# gets the same direction and candidates differ by seed only.
CANDIDATE_VARIANTS: Final[tuple[str, ...]] = (
    "Draw the most natural, direct continuation of what is already there.",
)


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """The user's request after the automatic-continuation rule is applied."""

    user_prompt: str | None
    text: str
    is_automatic: bool


def resolve_user_prompt(user_prompt: str | None) -> ResolvedPrompt:
    """Use the default continuation prompt when the user prompt is empty or whitespace."""
    if user_prompt is not None and not isinstance(user_prompt, str):
        raise ValueError("user_prompt must be a string or None")
    if user_prompt is None or not user_prompt.strip():
        return ResolvedPrompt(user_prompt, DEFAULT_CONTINUATION_PROMPT, True)
    return ResolvedPrompt(user_prompt, user_prompt.strip(), False)


def variant_instruction(candidate_index: int) -> str:
    """Direction for one candidate; indices past the list reuse the last variant."""
    return CANDIDATE_VARIANTS[min(max(candidate_index, 0), len(CANDIDATE_VARIANTS) - 1)]


def build_generation_prompt(prompt: ResolvedPrompt, candidate_index: int | None = None) -> str:
    """Return the full instruction for an initial candidate.

    ``candidate_index`` adds that candidate's variant direction; ``None`` omits it.
    """
    variant = "" if candidate_index is None else f"\n\n{variant_instruction(candidate_index)}"
    if prompt.is_automatic:
        return f"{prompt.text}{variant}\n\n{_GUIDANCE_DESCRIPTION}"
    return (
        f"Drawing request: {prompt.text}\n\n{DEFAULT_CONTINUATION_PROMPT}{variant}\n\n"
        f"{_GUIDANCE_DESCRIPTION}"
    )


def build_revision_prompt(
    prompt: ResolvedPrompt,
    problems: tuple[str, ...],
    revision_instruction: str,
    candidate_index: int | None = None,
) -> str:
    """Return the instruction for the single permitted revision of a candidate."""
    lines = [build_generation_prompt(prompt, candidate_index), "", "Revise the previous attempt."]
    if problems:
        lines.append("Problems found:")
        lines.extend(f"- {problem}" for problem in problems)
    if revision_instruction.strip():
        lines.append(f"Requested change: {revision_instruction.strip()}")
    lines.append(
        "The protected areas and printable region are fixed; do not try to draw "
        "over or closer to existing marks than the guidance allows."
    )
    return "\n".join(lines)


CRITIC_RESPONSE_SCHEMA: Final[dict[str, JSONValue]] = {
    "score": "number 0-10, overall quality of the continuation",
    "boundary_continuity_score": "number 0-10, how well new strokes meet existing ones given the gaps",
    "style_match_score": "number 0-10, line width and drawing style match",
    "geometry_plausibility_score": "number 0-10, plausible proportions and perspective",
    "problems": "list of short strings",
    "revision_instruction": "one instruction for the image model, or an empty string",
    "recommendation": "one of: accept, revise, reject",
}


def build_critic_prompt(prompt: ResolvedPrompt, validation_metrics: dict[str, JSONValue]) -> str:
    """Return the critic instruction for one sanitized preview."""
    compact_metrics = {
        key: validation_metrics[key]
        for key in ("final_ink_count", "removed_fraction", "connected_component_count")
        if key in validation_metrics
    }
    return "\n".join(
        [
            "You review a proposed continuation of a drawing on a physical page.",
            f"The drawing request was: {prompt.text}",
            "Images: (1) the original page, (2) region guidance where green is printable "
            "and red is forbidden, (3) the preview of the page with ONLY the ink that "
            "would actually be printed added in black, and, when present, (4) a close-up "
            "of image (3) around the printable region for judging stroke detail.",
            "Judge the preview, not an idealised version: small gaps where new strokes "
            "stop short of existing ink are required safety clearance and cannot be "
            "removed, but judge whether the continuation still reads naturally.",
            "Evaluate visible gaps, continuity, line width, plausible proportions, "
            "relevance to the drawing, and artifacts such as noise or stray marks.",
            "You cannot change the protected areas or the printable region.",
            f"Deterministic validation metrics: {json.dumps(compact_metrics, sort_keys=True)}",
            "Respond with only a JSON object with exactly these fields:",
            json.dumps(CRITIC_RESPONSE_SCHEMA, indent=2),
        ]
    )
