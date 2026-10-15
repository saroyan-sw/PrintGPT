"""Shared deterministic inputs for Track B tests (helpers only, no tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout
from printgpt_generation.config import GenerationConfig
from printgpt_generation.doubles import DeterministicCritic, DeterministicGenerator
from printgpt_generation.example import synthetic_page
from printgpt_generation.graph import GraphDependencies, run_generation
from printgpt_generation.providers import FixedStepClock, SequentialIdProvider
from printgpt_generation.requests import GenerationRequest, GenerationRunResult, create_generation_request
from printgpt_generation.track_a import CandidateValidator, SimulatedMaskBundle, SimulatedValidator, build_simulated_masks


# 75 DPI keeps rasters small (620 x 877) while 1 mm is still about 3 pixels.
LAYOUT = A4CalibrationLayout(dpi=75.0)


def make_config(**overrides: object) -> GenerationConfig:
    return GenerationConfig(layout=LAYOUT, **overrides)  # type: ignore[arg-type]


def make_page() -> tuple[np.ndarray, np.ndarray]:
    return synthetic_page(LAYOUT)


def make_masks(page: np.ndarray | None = None) -> SimulatedMaskBundle:
    source, allowed = make_page()
    return build_simulated_masks(source if page is None else page, allowed, layout=LAYOUT)


def make_request(user_prompt: str | None = "", *, seed: int = 3, config: GenerationConfig | None = None) -> GenerationRequest:
    page, allowed = make_page()
    masks = build_simulated_masks(page, allowed, layout=LAYOUT)
    return create_generation_request(page, masks, user_prompt, config=config or make_config(), seed=seed)


@dataclass
class Run:
    result: GenerationRunResult
    generator: DeterministicGenerator
    critic: DeterministicCritic


def make_deps(
    generator: DeterministicGenerator | None = None,
    critic: DeterministicCritic | None = None,
    *,
    validator: CandidateValidator | None = None,
    config: GenerationConfig | None = None,
) -> GraphDependencies:
    return GraphDependencies(
        generator=generator or DeterministicGenerator(),
        validator=validator or SimulatedValidator(),
        critic=critic or DeterministicCritic(),
        clock=FixedStepClock(),
        ids=SequentialIdProvider(),
        config=config or make_config(),
    )


def run_workflow(
    generator: DeterministicGenerator | None = None,
    critic: DeterministicCritic | None = None,
    *,
    user_prompt: str | None = "",
    seed: int = 3,
    config: GenerationConfig | None = None,
) -> Run:
    generator = generator or DeterministicGenerator()
    critic = critic or DeterministicCritic()
    config = config or make_config()
    request = make_request(user_prompt, seed=seed, config=config)
    result = run_generation(request, make_deps(generator, critic, config=config))
    return Run(result, generator, critic)
