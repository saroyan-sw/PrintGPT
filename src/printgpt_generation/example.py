"""Run the full offline workflow on a synthetic page with deterministic doubles.

Uses Track A's ``printgpt_masking`` when it is installed and the simulated
validator otherwise; the run record states which one produced the masks.
"""

from __future__ import annotations

from argparse import ArgumentParser
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np

from printgpt_alignment.geometry import A4CalibrationLayout, DEFAULT_LAYOUT, mm_to_px

from .config import GenerationConfig
from .doubles import CriticScript, DeterministicCritic, DeterministicGenerator, GeneratorBehavior
from .graph import GraphDependencies, run_generation
from .metadata import write_run_record
from .providers import FixedStepClock, SequentialIdProvider
from .requests import create_generation_request
from .track_a import CandidateValidator, MaskBundleLike, SimulatedValidator, TrackAValidator, build_simulated_masks


def _px(value_mm: float, layout: A4CalibrationLayout) -> int:
    return int(round(mm_to_px(value_mm, layout.dpi)))


def synthetic_page(layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> tuple[np.ndarray, np.ndarray]:
    """Return an unfinished line drawing and a central allowed region."""
    page = np.full((*layout.image_shape, 3), 255, dtype=np.uint8)
    thickness = max(2, _px(0.6, layout))
    cv2.ellipse(page, (_px(80, layout), _px(120, layout)), (_px(30, layout), _px(22, layout)), 0, 20, 300, (20, 20, 20), thickness)
    cv2.line(page, (_px(60, layout), _px(150, layout)), (_px(95, layout), _px(150, layout)), (20, 20, 20), thickness)
    allowed = np.zeros(layout.image_shape, dtype=np.uint8)
    allowed[_px(60, layout) : _px(240, layout), _px(40, layout) : _px(170, layout)] = 255
    return page, allowed


def _masks_and_validator(
    page: np.ndarray, allowed: np.ndarray, layout: A4CalibrationLayout
) -> tuple[MaskBundleLike, CandidateValidator]:
    if find_spec("printgpt_masking") is not None:
        from printgpt_masking import build_masks

        return build_masks(page, allowed, layout=layout), TrackAValidator()
    return build_simulated_masks(page, allowed, layout=layout), SimulatedValidator()


def run_example(output_dir: Path, layout: A4CalibrationLayout = DEFAULT_LAYOUT) -> Path:
    page, allowed = synthetic_page(layout)
    masks, validator = _masks_and_validator(page, allowed, layout)
    config = GenerationConfig(layout=layout)
    deps = GraphDependencies(
        # Candidate 2 starts outside the region so the run shows a revision.
        generator=DeterministicGenerator({(1, 0): GeneratorBehavior.OUTSIDE_REGION}),
        validator=validator,
        critic=DeterministicCritic({(0, 0): CriticScript.accept(7.5), (1, 1): CriticScript.accept(8.5)}),
        clock=FixedStepClock(),
        ids=SequentialIdProvider(),
        config=config,
    )
    request = create_generation_request(page, masks, "", config=config, seed=7)
    return write_run_record(run_generation(request, deps), output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("artifacts/generation_example"))
    args = parser.parse_args(argv)
    print(run_example(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
