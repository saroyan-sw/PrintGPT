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
from printgpt_generation.gemini import (  # noqa: E402
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


def _px(value_mm: float, scale: float) -> int:
    return int(round(mm_to_px(value_mm, LAYOUT.dpi) * scale))


def _decode(part: dict) -> np.ndarray:
    pixels = cv2.imdecode(np.frombuffer(base64.b64decode(part["data"]), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)


def _image_reply(body: dict, size: tuple[int, int]) -> dict:
    """Return the padded source at ``size`` (width, height) with one new stroke."""
    source = _decode(body["input"][0]["content"][1])
    output = cv2.resize(source, size, interpolation=cv2.INTER_LINEAR)
    scale = size[1] / source.shape[0]
    cv2.line(output, (_px(100, scale), _px(200, scale)), (_px(150, scale), _px(215, scale)), (0, 0, 0), 3)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    assert ok
    image = {"type": "image", "mime_type": "image/png", "data": base64.b64encode(encoded.tobytes()).decode()}
    return {"id": "img", "status": "completed", "steps": [{"type": "model_output", "content": [image]}],
            "usage": {"total_input_tokens": 1200, "total_output_tokens": 1300}}


def _critic_reply(text: str) -> dict:
    return {"id": "crit", "status": "completed",
            "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}]}


class FakeGemini:
    def __init__(self, *, image_size=(768, 1024), critic_text=json.dumps(CRITIC_JSON), image_status=200):
        self.bodies: list[dict] = []
        self.image_size = image_size
        self.critic_text = critic_text
        self.image_status = image_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body["response_format"]["type"] == "image":
            if self.image_status != 200:
                return httpx.Response(self.image_status, json={"error": {"code": self.image_status, "message": "x"}})
            return httpx.Response(200, json=_image_reply(body, self.image_size))
        return httpx.Response(200, json=_critic_reply(self.critic_text))


def _run(fake: FakeGemini):
    client = create_gemini_client("test-key", httpx_client=httpx.Client(transport=httpx.MockTransport(fake)))
    settings = GeminiSettings(image_model="image-model-x", critic_model="critic-model-y")
    deps = make_deps(GeminiImageGenerator(client, settings), GeminiVisualCritic(client, settings))
    return run_generation(make_request(), deps)


def test_gemini_adapters_complete_the_offline_workflow() -> None:
    fake = FakeGemini()
    result = _run(fake)
    assert result.outcome.status is RunStatus.CANDIDATES_READY
    assert result.generator_model_id == "image-model-x"
    image_bodies = [body for body in fake.bodies if body["response_format"]["type"] == "image"]
    critic_bodies = [body for body in fake.bodies if body["response_format"]["type"] == "text"]
    assert len(image_bodies) == 2 and len(critic_bodies) == 2
    assert image_bodies[0]["model"] == "image-model-x"
    assert image_bodies[0]["response_format"]["aspect_ratio"] == "3:4"
    assert critic_bodies[0]["response_format"]["schema"]["title"] == "CriticReport"
    attempt = result.attempts[0]
    assert attempt.generation.output is not None
    assert attempt.generation.output.image.shape == (*LAYOUT.image_shape, 3)
    assert attempt.generation.output.usage.input_tokens == 1200
    assert attempt.generation.output.usage.cost_usd is None
    assert attempt.status is AttemptStatus.CRITICIZED


def test_inputs_are_padded_to_requested_aspect() -> None:
    fake = FakeGemini()
    _run(fake)
    first = fake.bodies[0]["input"][0]["content"]
    source, guidance = _decode(first[1]), _decode(first[2])
    assert source.shape[0] == LAYOUT.image_shape[0]
    assert abs(source.shape[1] / source.shape[0] - 0.75) < 0.01
    assert source.shape == guidance.shape


def test_unexpected_output_aspect_is_a_provider_failure() -> None:
    result = _run(FakeGemini(image_size=(1024, 1024)))
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
    image_requests = [body for body in fake.bodies if body["response_format"]["type"] == "image"]
    # Two candidates x RetryPolicy.max_attempts_per_call; SDK-internal retries are disabled.
    assert len(image_requests) == 4
    assert result.outcome.status is RunStatus.NO_VALID_CANDIDATE


def test_malformed_gemini_critic_text_is_rejected() -> None:
    result = _run(FakeGemini(critic_text="Looks great!"))
    attempt = result.attempts[0]
    assert attempt.status is AttemptStatus.CRITIC_FAILED
    assert attempt.critic is not None and attempt.critic.failure is not None
    assert attempt.critic.failure.kind is ProviderFailureKind.MALFORMED_OUTPUT


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
