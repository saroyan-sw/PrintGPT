import cv2
import numpy as np

from printgpt_generation.doubles import DeterministicGenerator, GeneratorBehavior
from printgpt_generation.extraction import ExtractionConfig, NormalizationOutcome, normalize_and_extract
from printgpt_generation.guidance import OUTLINE_RGB, render_region_guidance
from printgpt_generation.models import AttemptKind, GeneratedImage
from printgpt_generation.providers import GenerationCall

from test_generation_fixtures import LAYOUT, make_masks, make_page


CONFIG = ExtractionConfig()


def _generate(behavior: GeneratorBehavior, seed: int = 5) -> tuple[GeneratedImage, np.ndarray]:
    page, _ = make_page()
    masks = make_masks(page)
    guidance = render_region_guidance(page, masks.printable_region_mask)
    call = GenerationCall(
        candidate_id="candidate-1",
        candidate_index=0,
        attempt_index=0,
        kind=AttemptKind.INITIAL,
        prompt="test",
        seed=seed,
        source_page=page,
        guidance_image=guidance,
        printable_region_mask=masks.printable_region_mask,
    )
    return DeterministicGenerator(default=behavior).generate(call), masks.existing_ink_mask


def _normalize(behavior: GeneratorBehavior) -> NormalizationOutcome:
    output, existing = _generate(behavior)
    return normalize_and_extract(output.image, existing, LAYOUT, CONFIG, output.additions_mask)


def _component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(mask, connectivity=8)
    return count - 1


def test_extracted_mask_follows_canonical_conventions() -> None:
    outcome = _normalize(GeneratorBehavior.CONTINUE)
    assert outcome.accepted
    assert outcome.candidate is not None
    mask = outcome.candidate.candidate_new_ink_mask
    assert mask.dtype == np.uint8
    assert mask.shape == LAYOUT.image_shape
    assert set(np.unique(mask).tolist()) == {0, 255}
    assert outcome.candidate.candidate_rgb.shape == (*LAYOUT.image_shape, 3)
    assert not mask.flags.writeable


def test_new_ink_never_includes_original_strokes() -> None:
    _, existing = _generate(GeneratorBehavior.CONTINUE)
    outcome = _normalize(GeneratorBehavior.CONTINUE)
    assert outcome.candidate is not None
    assert not np.any((outcome.candidate.candidate_new_ink_mask > 0) & (existing > 0))


def test_shifted_source_content_is_rejected() -> None:
    outcome = _normalize(GeneratorBehavior.SHIFT_SOURCE)
    assert not outcome.accepted
    assert outcome.record.registration is not None
    assert outcome.record.registration.estimated_shift_mm > CONFIG.max_registration_shift_mm
    assert any("moved" in reason for reason in outcome.record.failure_reasons)


def test_redrawn_original_strokes_are_not_reprinted() -> None:
    _, existing = _generate(GeneratorBehavior.REDRAW_SOURCE)
    clean = _normalize(GeneratorBehavior.CONTINUE)
    redrawn = _normalize(GeneratorBehavior.REDRAW_SOURCE)
    assert redrawn.accepted, redrawn.record.failure_reasons
    assert redrawn.candidate is not None and clean.candidate is not None
    assert redrawn.record.extraction is not None
    assert redrawn.record.extraction.discarded_source_redraw_pixels > int(np.count_nonzero(existing))
    # Only the genuinely new stroke survives; the thickened, offset originals do not.
    assert np.array_equal(redrawn.candidate.candidate_new_ink_mask, clean.candidate.candidate_new_ink_mask)


def test_background_noise_is_not_extracted_as_new_ink() -> None:
    noisy = _normalize(GeneratorBehavior.NOISY)
    assert noisy.accepted
    assert noisy.candidate is not None and noisy.record.extraction is not None
    assert noisy.record.extraction.discarded_noise_pixels > 0
    assert _component_count(noisy.candidate.candidate_new_ink_mask) == 1


def test_echoed_guidance_colours_never_become_ink() -> None:
    output, _ = _generate(GeneratorBehavior.GUIDANCE_ECHO)
    outline = np.all(output.image == np.asarray(OUTLINE_RGB, dtype=np.uint8), axis=2)
    assert outline.any()
    outcome = _normalize(GeneratorBehavior.GUIDANCE_ECHO)
    assert outcome.candidate is not None
    assert not np.any(outcome.candidate.candidate_new_ink_mask[outline])


def test_unsupported_aspect_ratio_is_rejected() -> None:
    outcome = _normalize(GeneratorBehavior.WRONG_ASPECT)
    assert not outcome.accepted
    assert "aspect ratio" in outcome.record.failure_reasons[0]


def test_uniformly_scaled_output_is_mapped_with_recorded_transform() -> None:
    output, _ = _generate(GeneratorBehavior.HALF_SCALE)
    outcome = _normalize(GeneratorBehavior.HALF_SCALE)
    assert outcome.accepted, outcome.record.failure_reasons
    scale_x = outcome.record.raw_to_canonical[0][0]
    assert scale_x == LAYOUT.image_shape[1] / output.image.shape[1]
    assert outcome.record.input_shape == output.image.shape


def test_non_uint8_output_is_rejected() -> None:
    _, existing = _generate(GeneratorBehavior.CONTINUE)
    outcome = normalize_and_extract(np.zeros((*LAYOUT.image_shape, 3), np.float32), existing, LAYOUT, CONFIG)
    assert not outcome.accepted


def test_explicit_additions_layer_is_preferred() -> None:
    outcome = _normalize(GeneratorBehavior.ADDITIONS_LAYER)
    assert outcome.record.extraction is not None
    assert outcome.record.extraction.method == "explicit_additions_layer"
    assert outcome.candidate is not None and outcome.candidate.candidate_new_ink_mask.any()


def test_blank_source_records_unverified_registration() -> None:
    blank_existing = np.zeros(LAYOUT.image_shape, np.uint8)
    page = np.full((*LAYOUT.image_shape, 3), 255, np.uint8)
    outcome = normalize_and_extract(page, blank_existing, LAYOUT, CONFIG)
    assert outcome.accepted
    assert outcome.record.registration is not None
    assert not outcome.record.registration.verified
