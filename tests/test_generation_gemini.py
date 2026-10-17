"""Gemini adapter tests against a mock HTTP transport; no network or API key.

The live smoke test runs only with ``PRINTGPT_LIVE_GEMINI=1`` and a key in
``GEMINI_API_KEY``; it makes paid requests.
"""

import base64
import json
import os

import cv2
import numpy as np
import pytest

pytest.importorskip("google.genai")
import httpx  # noqa: E402

from printgpt_alignment.geometry import mm_to_px  # noqa: E402
from printgpt_generation.framing import parse_aspect  # noqa: E402
from printgpt_generation.gemini import (  # noqa: E402
    CROP_ASPECTS,
    OUTPUT_MIME_TYPE,
    GeminiImageGenerator,
    GeminiSettings,
    GeminiVisualCritic,
    create_gemini_client,
)
from printgpt_generation.graph import run_generation  # noqa: E402
from printgpt_generation.models import AttemptStatus, ProviderFailureKind, RunStatus  # noqa: E402

from test_generation_fixtures import LAYOUT, make_deps, make_request  # noqa: E402


CRITIC_JSON = {
    "score": 8,
    "boundary_continuity_score": 7,
    "style_match_score": 7,
    "geometry_plausibility_score": 7,
    "problems": [],
    "revision_instruction": "",
    "recommendation": "accept",
}


def _decode(part: dict) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(part["data"]), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def _requested_size(body: dict, long_side: int = 1024) -> tuple[int, int]:
    ratio = parse_aspect(body["response_format"]["aspect_ratio"])
    if ratio >= 1:
        return long_side, int(round(long_side / ratio))
    return int(round(long_side * ratio)), long_side


def _image_reply(body: dict, size: tuple[int, int] | None, shift_mm: float) -> dict:
    """Return the sent window at the requested size with one stroke in its green region."""
    content = body["input"][0]["content"]
    source, guidance = _decode(content[1]), _decode(content[2])
    size = size or _requested_size(body)
    output = cv2.resize(source, size, interpolation=cv2.INTER_LINEAR)
    guidance = cv2.resize(guidance, size, interpolation=cv2.INTER_NEAREST)
    green = (guidance[:, :, 1].astype(int) - guidance[:, :, 0].astype(int)) > 15
    ys, xs = np.nonzero(green)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    start = (int(x0 + 0.45 * (x1 - x0)), int(y0 + 0.75 * (y1 - y0)))
    end = (int(x0 + 0.85 * (x1 - x0)), int(y0 + 0.80 * (y1 - y0)))
    cv2.line(output, start, end, (0, 0, 0), 3)
    if shift_mm:
        shift = mm_to_px(shift_mm, LAYOUT.dpi) * size[1] / source.shape[0]
        output = cv2.warpAffine(output, np.float32([[1, 0, shift], [0, 1, 0]]), size, borderValue=(255, 255, 255))
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    image = {"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(encoded.tobytes()).decode()}
    return {"id": "img", "status": "completed", "steps": [{"type": "model_output", "content": [image]}],
            "usage": {"total_input_tokens": 1200, "total_output_tokens": 1300}}


def _critic_reply(text: str) -> dict:
    return {"id": "crit", "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}


class FakeGemini:
    def __init__(self, *, image_size=None, critic_text=json.dumps(CRITIC_JSON), image_status=200, shift_mm=0.0):
        self.bodies: list[dict] = []
        self.image_size = image_size
        self.critic_text = critic_text
        self.image_status = image_status
        self.shift_mm = shift_mm

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body["response_format"]["type"] == "image":
            if self.image_status != 200:
                return httpx.Response(self.image_status, json={"error": {"code": self.image_status, "message": "x"}})
            return httpx.Response(200, json=_image_reply(body, self.image_size, self.shift_mm))
        return httpx.Response(200, json=_critic_reply(self.critic_text))

    def image_bodies(self) -> list[dict]:
        return [body for body in self.bodies if body["response_format"]["type"] == "image"]

    def critic_bodies(self) -> list[dict]:
        return [body for body in self.bodies if body["response_format"]["type"] == "text"]


def _run(fake: FakeGemini, **settings: object):
    client = create_gemini_client("test-key", httpx_client=httpx.Client(transport=httpx.MockTransport(fake)))
    gemini_settings = GeminiSettings(image_model="image-model-x", critic_model="critic-model-y", **settings)
    deps = make_deps(GeminiImageGenerator(client, gemini_settings), GeminiVisualCritic(client, gemini_settings))
    return run_generation(make_request(), deps)


def test_gemini_adapters_complete_the_workflow_with_crop_framing() -> None:
    fake = FakeGemini()
    result = _run(fake)
    assert result.outcome.status is RunStatus.CANDIDATES_READY
    assert result.generator_model_id == "image-model-x"
    image_bodies, critic_bodies = fake.image_bodies(), fake.critic_bodies()
    assert len(image_bodies) == 2 and len(critic_bodies) == 2
    assert image_bodies[0]["model"] == "image-model-x"
    assert image_bodies[0]["response_format"]["aspect_ratio"] in CROP_ASPECTS
    assert image_bodies[0]["response_format"]["mime_type"] == OUTPUT_MIME_TYPE
    assert critic_bodies[0]["response_format"]["schema"]["title"] == "CriticReport"
    attempt = result.attempts[0]
    output = attempt.generation.output
    assert output is not None and output.canvas_box is not None
    x, y, width, height = output.canvas_box
    assert output.image.shape == (height, width, 3)
    assert (height, width) != LAYOUT.image_shape
    assert attempt.normalization is not None and attempt.normalization.canvas_box == output.canvas_box
    assert attempt.normalization.registration is not None and attempt.normalization.registration.verified
    assert attempt.normalization.raw_to_canonical[0][2] == x and attempt.normalization.raw_to_canonical[1][2] == y
    assert attempt.normalized is not None and attempt.normalized.candidate_rgb.shape == (*LAYOUT.image_shape, 3)
    assert output.usage.input_tokens == 1200 and output.usage.cost_usd is None
    assert attempt.status is AttemptStatus.CRITICIZED


def test_crop_contains_the_printable_region_and_a_context_thumbnail() -> None:
    fake = FakeGemini()
    result = _run(fake)
    request = result.request
    region = np.argwhere(request.masks.printable_region_mask > 0)
    x, y, width, height = result.attempts[0].generation.output.canvas_box
    assert region[:, 0].min() >= y and region[:, 0].max() < y + height
    assert region[:, 1].min() >= x and region[:, 1].max() < x + width
    content = fake.image_bodies()[0]["input"][0]["content"]
    sent = _decode(content[1])
    ratio = parse_aspect(fake.image_bodies()[0]["response_format"]["aspect_ratio"])
    assert abs(sent.shape[1] / sent.shape[0] / ratio - 1) < 0.01
    assert sent.shape == _decode(content[2]).shape
    assert "Context only" in content[3]["text"]
    assert max(_decode(content[4]).shape[:2]) <= 1024


def test_candidates_get_distinct_variants_and_seeds() -> None:
    fake = FakeGemini()
    _run(fake)
    first, second = fake.image_bodies()
    assert "Variant A" in first["input"][0]["content"][0]["text"]
    assert "Variant B" in second["input"][0]["content"][0]["text"]
    assert first["generation_config"]["seed"] != second["generation_config"]["seed"]


def test_critic_receives_a_close_up_of_the_region() -> None:
    fake = FakeGemini()
    _run(fake)
    content = fake.critic_bodies()[0]["input"][0]["content"]
    assert len(content) == 5
    page, close_up = _decode(content[1]), _decode(content[4])
    assert close_up.shape[0] < page.shape[0]


def test_evenly_shifted_crop_is_aligned_back() -> None:
    result = _run(FakeGemini(shift_mm=3.0))
    attempt = result.attempts[0]
    assert attempt.status is AttemptStatus.CRITICIZED
    assert attempt.normalization is not None and attempt.normalization.alignment is not None
    assert attempt.normalization.alignment.applied


def test_shift_beyond_the_alignment_limit_is_rejected_inside_the_window() -> None:
    result = _run(FakeGemini(shift_mm=8.0))
    attempt = result.attempts[0]
    assert attempt.status is AttemptStatus.GEOMETRY_REJECTED
    assert attempt.normalization is not None
    assert any("moved" in reason or "preserved" in reason for reason in attempt.normalization.failure_reasons)


def test_full_page_framing_pads_inputs_to_three_by_four() -> None:
    fake = FakeGemini()
    result = _run(fake, framing="full_page")
    first = fake.image_bodies()[0]
    assert first["response_format"]["aspect_ratio"] == "3:4"
    content = first["input"][0]["content"]
    source, guidance = _decode(content[1]), _decode(content[2])
    assert source.shape[0] == LAYOUT.image_shape[0]
    assert abs(source.shape[1] / source.shape[0] - 0.75) < 0.01
    assert source.shape == guidance.shape
    assert len(content) == 3
    output = result.attempts[0].generation.output
    assert output is not None and output.canvas_box is None
    assert output.image.shape == (*LAYOUT.image_shape, 3)
    assert result.outcome.status is RunStatus.CANDIDATES_READY


def test_unexpected_output_aspect_is_a_provider_failure() -> None:
    result = _run(FakeGemini(image_size=(1024, 1024)), framing="full_page")
    attempt = result.attempts[0]
    assert attempt.status is AttemptStatus.GENERATION_FAILED
    assert attempt.generation.failure is not None
    assert "not the requested 3:4" in attempt.generation.failure.message
    assert attempt.generation.provider_attempts == 1


def test_rate_limit_is_retried_by_policy_only() -> None:
    fake = FakeGemini(image_status=429)
    result = _run(fake)
    attempt = result.attempts[0]
    assert attempt.generation.failure is not None
    assert attempt.generation.failure.kind is ProviderFailureKind.ERROR
    # Two candidates x RetryPolicy.max_attempts_per_call; SDK-internal retries are disabled.
    assert len(fake.image_bodies()) == 4
    assert result.outcome.status is RunStatus.NO_VALID_CANDIDATE


def test_malformed_gemini_critic_text_is_rejected() -> None:
    result = _run(FakeGemini(critic_text="Looks great!"))
    attempt = result.attempts[0]
    assert attempt.status is AttemptStatus.CRITIC_FAILED
    assert attempt.critic is not None and attempt.critic.failure is not None
    assert attempt.critic.failure.kind is ProviderFailureKind.MALFORMED_OUTPUT


def test_settings_reject_unknown_framing_and_aspects() -> None:
    with pytest.raises(ValueError):
        GeminiSettings(framing="tiles")
    with pytest.raises(ValueError):
        GeminiSettings(crop_aspects=("7:3",))


@pytest.mark.skipif(
    os.environ.get("PRINTGPT_LIVE_GEMINI") != "1" or not os.environ.get("GEMINI_API_KEY"),
    reason="live Gemini test needs PRINTGPT_LIVE_GEMINI=1 and GEMINI_API_KEY",
)
def test_live_gemini_smoke() -> None:
    client = create_gemini_client()
    deps = make_deps(GeminiImageGenerator(client), GeminiVisualCritic(client))
    result = run_generation(make_request(), deps)
    assert result.outcome.status in (RunStatus.CANDIDATES_READY, RunStatus.NO_VALID_CANDIDATE)
    assert all(attempt.generation.provider_attempts >= 1 for attempt in result.attempts)
