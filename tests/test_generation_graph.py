from dataclasses import replace

import numpy as np
import pytest
from langgraph.errors import GraphRecursionError

from printgpt_generation.doubles import (
    CriticBehavior,
    CriticScript,
    DeterministicCritic,
    DeterministicGenerator,
    GeneratorBehavior,
)
from printgpt_generation.graph import build_generation_graph, initial_state, run_generation
from printgpt_generation.models import (
    AttemptKind,
    AttemptStatus,
    ProviderFailureKind,
    RevisionDecisionKind,
    RunStatus,
)
from printgpt_generation.prompts import DEFAULT_CONTINUATION_PROMPT
from printgpt_generation.providers import RetryPolicy
from printgpt_generation.ranking import RevisionPolicy
from printgpt_generation.requests import create_generation_request
from printgpt_generation.track_a import compose_black_preview

from test_generation_fixtures import make_config, make_deps, make_masks, make_request, run_workflow


ALWAYS_REVISE = DeterministicCritic(default=CriticScript.revise())
GOOD_REVISE = CriticScript(recommendation="revise", score=7.0, sub_score=6.0, revision_instruction="Smoother curve.")


def _statuses(run) -> dict[tuple[int, int], AttemptStatus]:
    return {(attempt.candidate_index, attempt.attempt_index): attempt.status for attempt in run.result.attempts}


def test_empty_prompt_runs_automatic_continuation() -> None:
    run = run_workflow(user_prompt="   ")
    assert run.result.prompt.is_automatic
    assert all(call.prompt.startswith(DEFAULT_CONTINUATION_PROMPT) for call in run.generator.calls)


def test_two_initial_candidates_are_generated() -> None:
    run = run_workflow()
    initial = [call for call in run.generator.calls if call.kind is AttemptKind.INITIAL]
    assert [(call.candidate_index, call.attempt_index) for call in initial] == [(0, 0), (1, 0)]
    assert len(set(call.candidate_id for call in initial)) == 2
    assert initial[0].seed != initial[1].seed
    assert run.result.outcome.status is RunStatus.CANDIDATES_READY
    assert len(run.result.outcome.ranked) == 2


def test_source_page_is_preserved_and_raw_output_kept_separately() -> None:
    request = make_request()
    before = request.canonical_page.copy()
    generator = DeterministicGenerator(default=GeneratorBehavior.NOISY)
    result = run_generation(request, make_deps(generator))
    assert np.array_equal(request.canonical_page, before)
    for attempt in result.attempts:
        assert attempt.generation.output is not None and attempt.preview_rgb is not None
        assert attempt.sanitized_new_ink_mask is not None
        untouched = attempt.sanitized_new_ink_mask == 0
        assert np.array_equal(attempt.preview_rgb[untouched], before[untouched])
        assert not np.array_equal(attempt.generation.output.image, attempt.preview_rgb)


def test_critic_receives_sanitized_preview_not_raw_output() -> None:
    run = run_workflow(DeterministicGenerator(default=GeneratorBehavior.NOISY))
    assert len(run.critic.calls) == 2
    for call in run.critic.calls:
        attempt = next(
            a for a in run.result.attempts
            if (a.candidate_index, a.attempt_index) == (call.candidate_index, call.attempt_index)
        )
        assert attempt.sanitized_new_ink_mask is not None and attempt.generation.output is not None
        expected = compose_black_preview(run.result.request.canonical_page, attempt.sanitized_new_ink_mask)
        assert np.array_equal(call.sanitized_preview, expected)
        assert not np.array_equal(call.sanitized_preview, attempt.generation.output.image)
        assert np.array_equal(call.original_page, run.result.request.canonical_page)
        assert np.array_equal(call.guidance_image, run.result.guidance_image)


def test_validation_rejection_skips_critic_and_triggers_revision() -> None:
    generator = DeterministicGenerator({(0, 0): GeneratorBehavior.OUTSIDE_REGION})
    run = run_workflow(generator)
    original = run.result.attempt("candidate-1/attempt-0")
    assert original.status is AttemptStatus.VALIDATION_REJECTED
    assert original.critic is None
    assert original.revision_decision is not None
    assert original.revision_decision.kind is RevisionDecisionKind.REVISE
    assert original.validation is not None and original.validation.reasons
    assert [(c.candidate_index, c.attempt_index) for c in run.critic.calls] == [(0, 1), (1, 0)]


def test_revision_is_normalized_validated_and_criticized_again() -> None:
    generator = DeterministicGenerator({(0, 0): GeneratorBehavior.OUTSIDE_REGION})
    run = run_workflow(generator)
    revision = run.result.attempt("candidate-1/attempt-1")
    assert revision.kind is AttemptKind.REVISION
    assert revision.revision_of == "candidate-1/attempt-0"
    assert revision.normalization is not None and revision.validation is not None
    assert revision.validation.accepted
    assert revision.status is AttemptStatus.CRITICIZED
    assert "Problems found" in revision.prompt
    selection = run.result.outcome.selections[0]
    assert selection.selected_attempt_id == "candidate-1/attempt-1"


def test_revision_with_unsupported_geometry_is_rejected_again() -> None:
    generator = DeterministicGenerator(
        {(0, 0): GeneratorBehavior.OUTSIDE_REGION, (0, 1): GeneratorBehavior.WRONG_ASPECT}
    )
    run = run_workflow(generator)
    revision = run.result.attempt("candidate-1/attempt-1")
    assert revision.status is AttemptStatus.GEOMETRY_REJECTED
    assert revision.revision_decision is not None
    assert revision.revision_decision.kind is RevisionDecisionKind.LIMIT_REACHED


def test_each_candidate_gets_at_most_one_revision() -> None:
    run = run_workflow(critic=DeterministicCritic(default=CriticScript.revise()))
    assert len(run.generator.calls) == 4
    assert len(run.critic.calls) == 4
    assert dict(run.result.revision_counts) == {"candidate-1": 1, "candidate-2": 1}
    assert sorted(_statuses(run)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert run.result.outcome.status is RunStatus.NO_VALID_CANDIDATE


def test_revision_can_be_disabled() -> None:
    config = make_config(revision=RevisionPolicy(max_revisions_per_candidate=0))
    run = run_workflow(critic=DeterministicCritic(default=CriticScript.revise()), config=config)
    assert len(run.generator.calls) == 2
    assert not run.result.revision_counts


def test_failed_revision_keeps_valid_original() -> None:
    generator = DeterministicGenerator({(0, 1): GeneratorBehavior.OUTSIDE_REGION})
    critic = DeterministicCritic({(0, 0): GOOD_REVISE})
    run = run_workflow(generator, critic)
    assert _statuses(run)[(0, 1)] is AttemptStatus.VALIDATION_REJECTED
    selection = run.result.outcome.selections[0]
    assert selection.selected_attempt_id == "candidate-1/attempt-0"
    assert "original kept" in selection.reason
    assert any(item.attempt_id == "candidate-1/attempt-0" for item in run.result.outcome.ranked)


def test_lower_scoring_revision_keeps_original() -> None:
    critic = DeterministicCritic({(0, 0): GOOD_REVISE, (0, 1): CriticScript.accept(6.5)})
    run = run_workflow(critic=critic)
    assert run.result.outcome.selections[0].selected_attempt_id == "candidate-1/attempt-0"


def test_ranking_includes_only_eligible_candidates() -> None:
    critic = DeterministicCritic({(0, 0): CriticScript.reject(), (0, 1): CriticScript.reject()})
    run = run_workflow(critic=critic)
    ranked = run.result.outcome.ranked
    assert [item.candidate_id for item in ranked] == ["candidate-2"]
    assert run.result.outcome.selections[0].selected_attempt_id is None


def test_ranking_orders_by_critic_score() -> None:
    critic = DeterministicCritic({(0, 0): CriticScript.accept(7.0), (1, 0): CriticScript.accept(9.0)})
    run = run_workflow(critic=critic)
    assert [item.candidate_id for item in run.result.outcome.ranked] == ["candidate-2", "candidate-1"]
    assert [item.rank for item in run.result.outcome.ranked] == [1, 2]


def test_no_valid_candidate_is_explicit() -> None:
    run = run_workflow(DeterministicGenerator(default=GeneratorBehavior.BLANK))
    outcome = run.result.outcome
    assert outcome.status is RunStatus.NO_VALID_CANDIDATE
    assert outcome.ranked == ()
    assert len(outcome.reasons) == 2
    assert outcome.requires_user_approval
    assert run.critic.calls == []


@pytest.mark.parametrize("behavior", [CriticBehavior.MALFORMED_JSON, CriticBehavior.SCHEMA_VIOLATION])
def test_malformed_critic_output_fails_candidate_without_retry(behavior: CriticBehavior) -> None:
    critic = DeterministicCritic({(0, 0): CriticScript(behavior=behavior)})
    run = run_workflow(critic=critic)
    attempt = run.result.attempt("candidate-1/attempt-0")
    assert attempt.status is AttemptStatus.CRITIC_FAILED
    assert attempt.critic is not None and attempt.critic.failure is not None
    assert attempt.critic.failure.kind is ProviderFailureKind.MALFORMED_OUTPUT
    assert attempt.critic.provider_attempts == 1
    assert attempt.revision_decision is not None
    assert attempt.revision_decision.kind is RevisionDecisionKind.NOT_REVISABLE
    assert [item.candidate_id for item in run.result.outcome.ranked] == ["candidate-2"]


def test_generator_timeout_is_retried_within_policy_then_recorded() -> None:
    run = run_workflow(DeterministicGenerator({(0, 0): GeneratorBehavior.TIMEOUT}))
    attempt = run.result.attempt("candidate-1/attempt-0")
    assert attempt.status is AttemptStatus.GENERATION_FAILED
    assert attempt.generation.failure is not None
    assert attempt.generation.failure.kind is ProviderFailureKind.TIMEOUT
    assert attempt.generation.provider_attempts == RetryPolicy().max_attempts_per_call
    assert attempt.revision_decision is not None
    assert attempt.revision_decision.kind is RevisionDecisionKind.NOT_REVISABLE
    assert run.result.revision_counts.get("candidate-1", 0) == 0


def test_permanent_provider_error_is_not_retried() -> None:
    run = run_workflow(DeterministicGenerator({(1, 0): GeneratorBehavior.ERROR}))
    attempt = run.result.attempt("candidate-2/attempt-0")
    assert attempt.status is AttemptStatus.GENERATION_FAILED
    assert attempt.generation.provider_attempts == 1
    assert [item.candidate_id for item in run.result.outcome.ranked] == ["candidate-1"]


def test_transient_failure_retry_does_not_count_as_revision() -> None:
    run = run_workflow(DeterministicGenerator(transient_failures={(0, 0): 1}))
    attempt = run.result.attempt("candidate-1/attempt-0")
    assert attempt.status is AttemptStatus.CRITICIZED
    assert attempt.generation.provider_attempts == 2
    assert not run.result.revision_counts


def test_critic_timeout_is_recorded() -> None:
    run = run_workflow(critic=DeterministicCritic({(1, 0): CriticScript(behavior=CriticBehavior.TIMEOUT)}))
    attempt = run.result.attempt("candidate-2/attempt-0")
    assert attempt.status is AttemptStatus.CRITIC_FAILED
    assert attempt.critic is not None and attempt.critic.failure is not None
    assert attempt.critic.failure.kind is ProviderFailureKind.TIMEOUT


def test_total_provider_attempts_are_bounded_across_the_run() -> None:
    config = make_config(retry=RetryPolicy(max_attempts_per_call=3, max_total_attempts=4))
    run = run_workflow(DeterministicGenerator(default=GeneratorBehavior.TIMEOUT), config=config)
    assert len(run.generator.calls) == 4
    assert run.result.counters.provider_attempts == 4
    last = run.result.attempt("candidate-2/attempt-0")
    assert last.generation.failure is not None
    assert last.generation.failure.kind is ProviderFailureKind.TIMEOUT
    assert run.result.outcome.status is RunStatus.NO_VALID_CANDIDATE


def test_exhausted_budget_is_a_typed_failure() -> None:
    config = make_config(retry=RetryPolicy(max_attempts_per_call=2, max_total_attempts=2))
    run = run_workflow(DeterministicGenerator(default=GeneratorBehavior.TIMEOUT), config=config)
    last = run.result.attempt("candidate-2/attempt-0")
    assert last.generation.failure is not None
    assert last.generation.failure.kind is ProviderFailureKind.BUDGET_EXHAUSTED
    assert last.generation.provider_attempts == 0


def test_worst_case_node_and_call_counts_are_bounded() -> None:
    generator = DeterministicGenerator()
    critic = DeterministicCritic(default=CriticScript.revise())
    deps = make_deps(generator, critic)
    graph = build_generation_graph(deps)
    steps = list(
        graph.stream(
            initial_state(make_request()),
            stream_mode="updates",
            config={"recursion_limit": deps.config.recursion_limit},
        )
    )
    # prepare + generate + 2 x (select + 4 stages + revise + 4 stages) + final select + rank
    assert len(steps) == 24
    assert len(steps) < deps.config.recursion_limit
    assert len(generator.calls) == deps.config.max_generation_requests == 4
    assert len(critic.calls) <= 4


def test_recursion_limit_is_a_final_guard() -> None:
    deps = make_deps(critic=DeterministicCritic(default=CriticScript.revise()), config=make_config(recursion_limit=10))
    graph = build_generation_graph(deps)
    with pytest.raises(GraphRecursionError):
        graph.invoke(initial_state(make_request()), config={"recursion_limit": deps.config.recursion_limit})


def test_invalid_request_inputs_raise_value_error() -> None:
    masks = make_masks()
    with pytest.raises(ValueError, match="canonical_page"):
        create_generation_request(np.zeros((10, 10, 3), np.uint8), masks, config=make_config())
    page = np.full((*make_config().layout.image_shape, 3), 255, np.uint8)
    no_allowed_region = replace(masks, allowed_region_mask=np.zeros_like(masks.allowed_region_mask))
    with pytest.raises(ValueError, match="allowed_region_mask"):
        create_generation_request(page, no_allowed_region, config=make_config())
