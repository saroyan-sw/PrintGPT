"""Runs only when Track A's ``printgpt_masking`` is importable (after integration)."""

import numpy as np
import pytest

from printgpt_generation.doubles import DeterministicGenerator, GeneratorBehavior
from printgpt_generation.graph import run_generation
from printgpt_generation.models import AttemptStatus, RunStatus
from printgpt_generation.requests import create_generation_request
from printgpt_generation.track_a import TrackAValidator

from test_generation_fixtures import LAYOUT, make_config, make_deps, make_page

masking = pytest.importorskip("printgpt_masking")


def test_workflow_uses_track_a_masks_and_validator() -> None:
    page, allowed = make_page()
    masks = masking.build_masks(page, allowed, layout=LAYOUT)
    config = make_config()
    request = create_generation_request(page, masks, "", config=config, seed=4)
    generator = DeterministicGenerator({(1, 0): GeneratorBehavior.OUTSIDE_REGION})
    result = run_generation(request, make_deps(generator, validator=TrackAValidator(), config=config))

    assert not result.simulated_validation
    assert result.outcome.status is RunStatus.CANDIDATES_READY
    assert result.attempt("candidate-2/attempt-0").status is AttemptStatus.VALIDATION_REJECTED
    for attempt in result.attempts:
        if attempt.sanitized_new_ink_mask is not None:
            printable = masks.printable_region_mask > 0
            assert not np.any((attempt.sanitized_new_ink_mask > 0) & ~printable)
            assert attempt.validation is not None
            assert "removed_fraction" in attempt.validation.metrics
