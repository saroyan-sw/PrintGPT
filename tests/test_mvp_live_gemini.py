"""Live-Gemini backend selection in the MVP pipeline, with a mock transport only."""

import cv2
import pytest

pytest.importorskip("google.genai")
import httpx  # noqa: E402

from printgpt_alignment.calibration_page import render_calibration_image  # noqa: E402
from printgpt_alignment.geometry import A4CalibrationLayout, mm_to_px  # noqa: E402
from printgpt_generation.gemini import GeminiSettings, create_gemini_client  # noqa: E402
from printgpt_generation.models import RunStatus  # noqa: E402
from printgpt_mvp import InputMode, approve_and_export, rectangular_allowed_region, run_offline_document  # noqa: E402
from printgpt_mvp.pipeline import GenerationBackend, LiveGenerationUnavailable, gemini_api_key  # noqa: E402

from test_generation_gemini import FakeGemini  # noqa: E402


LAYOUT = A4CalibrationLayout(dpi=75.0)
SETTINGS = GeminiSettings(image_model="image-model-x", critic_model="critic-model-y")


def _px(value_mm: float) -> int:
    return int(round(mm_to_px(value_mm, LAYOUT.dpi)))


@pytest.fixture
def marked_page(tmp_path):
    page = render_calibration_image(LAYOUT)
    cv2.ellipse(page, (_px(80), _px(120)), (_px(30), _px(22)), 0, 20, 300, 20, 2)
    cv2.line(page, (_px(60), _px(150)), (_px(95), _px(150)), 20, 2)
    path = tmp_path / "unfinished.png"
    assert cv2.imwrite(str(path), page)
    return path


def _run(input_path, output_dir, backend, fake=None):
    client = None
    if fake is not None:
        client = create_gemini_client("test-key", httpx_client=httpx.Client(transport=httpx.MockTransport(fake)))
    return run_offline_document(
        input_path,
        rectangular_allowed_region(x_mm=40, y_mm=60, width_mm=130, height_mm=180, layout=LAYOUT),
        output_dir,
        seed=7,
        layout=LAYOUT,
        input_mode=InputMode.MARKED_PAGE,
        normalization_confirmed=True,
        backend=backend,
        gemini_settings=SETTINGS,
        gemini_client=client,
    )


def test_missing_key_fails_before_anything_is_written(marked_page, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    output = tmp_path / "live-run"
    with pytest.raises(LiveGenerationUnavailable, match="GEMINI_API_KEY"):
        _run(marked_page, output, GenerationBackend.GEMINI)
    assert not output.exists()


def test_key_is_read_from_environment_then_dotenv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    dotenv = tmp_path / ".env"
    assert gemini_api_key(dotenv) is None
    dotenv.write_text("GEMINI_API_KEY='from-dotenv'\n", encoding="utf-8")
    assert gemini_api_key(dotenv) == "from-dotenv"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert gemini_api_key(dotenv) == "from-env"


def test_live_backend_swaps_only_generator_and_critic(marked_page, tmp_path) -> None:
    fake = FakeGemini()
    run = _run(marked_page, tmp_path / "live-run", "gemini", fake)
    generation = run.generation
    assert generation.outcome.status is RunStatus.CANDIDATES_READY
    assert generation.generator_model_id == "image-model-x"
    assert generation.critic_model_id == "critic-model-y"
    assert not generation.simulated_validation  # TrackAValidator, not the simulated double
    assert generation.counters.generation_requests <= 4 and generation.counters.critic_requests <= 4
    assert all(candidate.attempt.generation.output.canvas_box is not None for candidate in run.candidates)
    assert all(candidate.attempt.validation.accepted for candidate in run.candidates)
    assert not (run.output_dir / "print").exists()


def test_failed_live_run_has_nothing_printable_and_offline_still_works(marked_page, tmp_path) -> None:
    output = tmp_path / "run"
    failed = _run(marked_page, output, GenerationBackend.GEMINI, FakeGemini(image_status=503))
    assert failed.generation.outcome.status is RunStatus.NO_VALID_CANDIDATE
    assert not failed.candidates
    with pytest.raises(ValueError):
        approve_and_export(failed, "candidate-1/attempt-0", approved=True)
    assert not (output / "print").exists()

    offline = _run(marked_page, output, GenerationBackend.OFFLINE)
    assert offline.generation.outcome.status is RunStatus.CANDIDATES_READY
    assert offline.generation.generator_model_id == "deterministic-generator-v1"
    assert len(offline.candidates) == 2
