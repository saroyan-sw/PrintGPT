import json
from datetime import UTC, datetime
from pathlib import Path

from printgpt_generation.doubles import DeterministicGenerator, GeneratorBehavior
from printgpt_generation.graph import GraphDependencies, run_generation
from printgpt_generation.metadata import build_run_metadata, write_run_record
from printgpt_generation.providers import FixedStepClock, UuidIdProvider
from printgpt_generation.track_a import SimulatedValidator
from printgpt_generation.doubles import DeterministicCritic

from test_generation_fixtures import make_config, make_request, run_workflow


def _decisions(metadata: dict) -> list[tuple]:
    return [
        (
            attempt["candidate_id"].split("-")[0],
            attempt["attempt_index"],
            attempt["status"],
            json.dumps(attempt["validation"], sort_keys=True),
            json.dumps(attempt["critic"]["verdict"] if attempt["critic"] else None, sort_keys=True),
            json.dumps(attempt["revision_decision"], sort_keys=True),
        )
        for attempt in metadata["attempts"]
    ]


def test_identical_seed_and_input_give_identical_metadata() -> None:
    generator = lambda: DeterministicGenerator({(1, 0): GeneratorBehavior.OUTSIDE_REGION})  # noqa: E731
    first = build_run_metadata(run_workflow(generator(), seed=11).result).model_dump(mode="json")
    second = build_run_metadata(run_workflow(generator(), seed=11).result).model_dump(mode="json")
    assert first == second


def test_decisions_do_not_depend_on_injected_time_or_ids() -> None:
    config = make_config()
    baseline = build_run_metadata(run_workflow(seed=11).result).model_dump(mode="json")
    deps = GraphDependencies(
        generator=DeterministicGenerator(),
        validator=SimulatedValidator(),
        critic=DeterministicCritic(),
        clock=FixedStepClock(start=datetime(2030, 5, 1, tzinfo=UTC)),
        ids=UuidIdProvider(),
        config=config,
    )
    other = build_run_metadata(run_generation(make_request(seed=11, config=config), deps)).model_dump(mode="json")
    assert other["run_id"] != baseline["run_id"]
    assert other["started_at"] != baseline["started_at"]
    assert _decisions(other) == _decisions(baseline)
    assert [item["score"] for item in other["ranking"]] == [item["score"] for item in baseline["ranking"]]


def test_run_record_is_compact_json_with_separate_artifacts(tmp_path: Path) -> None:
    run = run_workflow(DeterministicGenerator({(0, 0): GeneratorBehavior.OUTSIDE_REGION}))
    path = write_run_record(run.result, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert text.startswith('{\n  "schema_version"')
    record = json.loads(text)
    assert record["requires_user_approval"] is True
    assert record["approved"] is False
    assert record["automatic_prompt"] is True
    assert record["simulated_validation"] is True
    assert record["counters"] == {"generation_requests": 3, "critic_requests": 2, "provider_attempts": 5}
    for attempt in record["attempts"]:
        for relative in attempt["artifacts"].values():
            assert (tmp_path / relative).is_file()
    assert (tmp_path / record["artifacts"]["source_page"]).is_file()
    rejected = record["attempts"][0]
    assert rejected["status"] == "validation_rejected"
    assert rejected["validation"]["reasons"]
    assert rejected["normalization"]["raw_to_canonical"][2] == [0.0, 0.0, 1.0]
    # Usage is unknown for the doubles and must not be reported as zero.
    assert rejected["generation"]["usage"] == {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    assert path.stat().st_size < 64_000


def test_failure_information_is_recorded() -> None:
    run = run_workflow(DeterministicGenerator({(0, 0): GeneratorBehavior.TIMEOUT}))
    record = build_run_metadata(run.result).model_dump(mode="json")
    failed = record["attempts"][0]
    assert failed["status"] == "generation_failed"
    assert failed["generation"]["failure"] == {"kind": "timeout", "message": "simulated timeout"}
    assert failed["generation"]["provider_attempts"] == 2
