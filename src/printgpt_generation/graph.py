"""LangGraph orchestration of the bounded generation and criticism workflow.

    prepare_run -> generate_initial_candidates -> select_next_attempt
      -> normalize_candidate -> validate_candidate -> criticize_candidate
      -> decide_revision -> [generate_revision -> normalize ... -> decide_revision]
      -> select_next_attempt ... -> rank_candidates -> END

Candidates are processed one at a time from a queue. Each node performs one
transition and returns a partial state update built from new immutable values.
Routing reads only typed statuses. The one-revision rule is enforced by
``revision_counts`` in state and by ``ranking.decide_revision``; the
recursion limit is a final guard, not the mechanism that bounds the loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .config import GenerationConfig
from .critic import parse_critic_output, to_verdict
from .extraction import normalize_and_extract
from .guidance import render_region_guidance
from .models import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    CallCounters,
    CriticOutcome,
    GenerationOutcome,
    ProviderFailure,
    ProviderFailureKind,
    RGBImage,
    RevisionDecisionKind,
    RunOutcome,
    UsageRecord,
    readonly_copy,
)
from .prompts import (
    ResolvedPrompt,
    build_critic_prompt,
    build_generation_prompt,
    build_revision_prompt,
    resolve_user_prompt,
)
from .providers import (
    CandidateGenerator,
    Clock,
    CritiqueCall,
    GenerationCall,
    IdProvider,
    VisualCritic,
    call_with_retries,
)
from .ranking import build_outcome, decide_revision
from .requests import GenerationRequest, GenerationRunResult
from .track_a import CandidateValidator, summarize_validation


class GenerationGraphState(TypedDict):
    """Workflow state. Every field is replaced wholesale, never mutated in place."""

    request: GenerationRequest
    """Immutable input page, Track A masks, user prompt, and seed."""
    run_id: str
    """Injected identifier for this run (empty until ``prepare_run``)."""
    started_at: datetime | None
    finished_at: datetime | None
    prompt: ResolvedPrompt | None
    """User prompt after the automatic-continuation rule."""
    guidance_image: RGBImage | None
    """Region guidance shown to generator and critic; never printable."""
    candidate_ids: tuple[str, ...]
    attempts: tuple[AttemptRecord, ...]
    """All attempts in creation order, including failures."""
    pending_attempt_ids: tuple[str, ...]
    """Initial attempts not yet processed, in candidate order."""
    active_attempt_id: str | None
    """Attempt the current pass works on; ``None`` once the queue is empty."""
    revision_counts: Mapping[str, int]
    """Semantic revisions used per candidate ID (transport retries excluded)."""
    counters: CallCounters
    outcome: RunOutcome | None
    """Terminal result, set only by ``rank_candidates``."""


class StateUpdate(TypedDict, total=False):
    run_id: str
    started_at: datetime
    finished_at: datetime
    prompt: ResolvedPrompt
    guidance_image: RGBImage
    candidate_ids: tuple[str, ...]
    attempts: tuple[AttemptRecord, ...]
    pending_attempt_ids: tuple[str, ...]
    active_attempt_id: str | None
    revision_counts: Mapping[str, int]
    counters: CallCounters
    outcome: RunOutcome


def initial_state(request: GenerationRequest) -> GenerationGraphState:
    return GenerationGraphState(
        request=request,
        run_id="",
        started_at=None,
        finished_at=None,
        prompt=None,
        guidance_image=None,
        candidate_ids=(),
        attempts=(),
        pending_attempt_ids=(),
        active_attempt_id=None,
        revision_counts={},
        counters=CallCounters(),
        outcome=None,
    )


@dataclass(frozen=True, slots=True)
class GraphDependencies:
    """Everything the graph calls; swapping these switches doubles for live adapters."""

    generator: CandidateGenerator
    validator: CandidateValidator
    critic: VisualCritic
    clock: Clock
    ids: IdProvider
    config: GenerationConfig


# Node names double as route targets.
PREPARE: Final = "prepare_run"
GENERATE_INITIAL: Final = "generate_initial_candidates"
SELECT_NEXT: Final = "select_next_attempt"
NORMALIZE: Final = "normalize_candidate"
VALIDATE: Final = "validate_candidate"
CRITICIZE: Final = "criticize_candidate"
DECIDE_REVISION: Final = "decide_revision"
GENERATE_REVISION: Final = "generate_revision"
RANK: Final = "rank_candidates"

_REVISION_SEED_OFFSET: Final = 1000


def _active(state: GenerationGraphState) -> AttemptRecord:
    attempt_id = state["active_attempt_id"]
    for attempt in state["attempts"]:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise ValueError(f"active attempt {attempt_id!r} is not in state")


def _with_attempt(attempts: tuple[AttemptRecord, ...], updated: AttemptRecord) -> tuple[AttemptRecord, ...]:
    return tuple(updated if attempt.attempt_id == updated.attempt_id else attempt for attempt in attempts)


def _remaining_attempt_budget(counters: CallCounters, config: GenerationConfig) -> int:
    remaining = config.retry.max_total_attempts - counters.provider_attempts
    return max(0, min(config.retry.max_attempts_per_call, remaining))


class _Nodes:
    """Node implementations bound to injected dependencies."""

    def __init__(self, deps: GraphDependencies) -> None:
        self._deps = deps

    def _generate(self, call: GenerationCall, counters: CallCounters) -> tuple[AttemptRecord, CallCounters]:
        budget = _remaining_attempt_budget(counters, self._deps.config)
        result = call_with_retries(lambda: self._deps.generator.generate(call), max_attempts=budget, clock=self._deps.clock)
        outcome = GenerationOutcome(result.value, result.failure, result.attempts, result.latency_s)
        record = AttemptRecord(
            attempt_id=f"{call.candidate_id}/attempt-{call.attempt_index}",
            candidate_id=call.candidate_id,
            candidate_index=call.candidate_index,
            attempt_index=call.attempt_index,
            kind=call.kind,
            status=AttemptStatus.GENERATED if result.value is not None else AttemptStatus.GENERATION_FAILED,
            prompt=call.prompt,
            seed=call.seed,
            generation=outcome,
        )
        updated = replace(
            counters,
            generation_requests=counters.generation_requests + 1,
            provider_attempts=counters.provider_attempts + result.attempts,
        )
        return record, updated

    def prepare_run(self, state: GenerationGraphState) -> StateUpdate:
        request = state["request"]
        return StateUpdate(
            run_id=self._deps.ids.new_id("run"),
            started_at=self._deps.clock.now(),
            prompt=resolve_user_prompt(request.user_prompt),
            guidance_image=render_region_guidance(request.canonical_page, request.masks.printable_region_mask),
        )

    def generate_initial_candidates(self, state: GenerationGraphState) -> StateUpdate:
        request = state["request"]
        assert state["prompt"] is not None and state["guidance_image"] is not None
        prompt_text = build_generation_prompt(state["prompt"])
        counters = state["counters"]
        records: list[AttemptRecord] = []
        for index in range(self._deps.config.candidate_count):
            call = GenerationCall(
                candidate_id=self._deps.ids.new_id("candidate"),
                candidate_index=index,
                attempt_index=0,
                kind=AttemptKind.INITIAL,
                prompt=prompt_text,
                seed=request.seed + index,
                source_page=request.canonical_page,
                guidance_image=state["guidance_image"],
                printable_region_mask=request.masks.printable_region_mask,
            )
            record, counters = self._generate(call, counters)
            records.append(record)
        return StateUpdate(
            candidate_ids=tuple(record.candidate_id for record in records),
            attempts=tuple(records),
            pending_attempt_ids=tuple(record.attempt_id for record in records),
            counters=counters,
        )

    def select_next_attempt(self, state: GenerationGraphState) -> StateUpdate:
        pending = state["pending_attempt_ids"]
        if not pending:
            return StateUpdate(active_attempt_id=None)
        return StateUpdate(active_attempt_id=pending[0], pending_attempt_ids=pending[1:])

    def normalize_candidate(self, state: GenerationGraphState) -> StateUpdate:
        attempt = _active(state)
        assert attempt.generation.output is not None
        output = attempt.generation.output
        outcome = normalize_and_extract(
            output.image,
            state["request"].masks.existing_ink_mask,
            self._deps.config.layout,
            self._deps.config.extraction,
            output.additions_mask,
        )
        updated = replace(
            attempt,
            status=AttemptStatus.EXTRACTED if outcome.accepted else AttemptStatus.GEOMETRY_REJECTED,
            normalization=outcome.record,
            normalized=outcome.candidate,
        )
        return StateUpdate(attempts=_with_attempt(state["attempts"], updated))

    def validate_candidate(self, state: GenerationGraphState) -> StateUpdate:
        attempt = _active(state)
        assert attempt.normalized is not None
        request = state["request"]
        validator = self._deps.validator
        result = validator.validate(attempt.normalized.candidate_new_ink_mask, request.masks)
        sanitized = readonly_copy(result.sanitized_new_ink_mask)
        preview = readonly_copy(validator.compose_preview(request.canonical_page, sanitized))
        summary = summarize_validation(result, simulated=validator.simulated)
        updated = replace(
            attempt,
            status=AttemptStatus.VALIDATED if summary.accepted else AttemptStatus.VALIDATION_REJECTED,
            validation=summary,
            sanitized_new_ink_mask=sanitized,
            preview_rgb=preview,
        )
        return StateUpdate(attempts=_with_attempt(state["attempts"], updated))

    def criticize_candidate(self, state: GenerationGraphState) -> StateUpdate:
        attempt = _active(state)
        assert attempt.preview_rgb is not None and attempt.validation is not None
        assert state["prompt"] is not None and state["guidance_image"] is not None
        call = CritiqueCall(
            candidate_id=attempt.candidate_id,
            candidate_index=attempt.candidate_index,
            attempt_index=attempt.attempt_index,
            prompt=build_critic_prompt(state["prompt"], attempt.validation.metrics),
            original_page=state["request"].canonical_page,
            guidance_image=state["guidance_image"],
            sanitized_preview=attempt.preview_rgb,
        )
        critic = self._deps.critic
        result = call_with_retries(
            lambda: critic.critique(call),
            max_attempts=_remaining_attempt_budget(state["counters"], self._deps.config),
            clock=self._deps.clock,
        )
        failure = result.failure
        verdict = None
        usage = result.value.usage if result.value is not None else UsageRecord()
        if result.value is not None:
            parsed = parse_critic_output(result.value.text)
            if parsed.report is None:
                failure = ProviderFailure(ProviderFailureKind.MALFORMED_OUTPUT, parsed.error or "malformed")
            else:
                verdict = to_verdict(parsed.report, self._deps.config.critic)
        outcome = CriticOutcome(
            verdict=verdict,
            failure=failure,
            provider_attempts=result.attempts,
            latency_s=result.latency_s,
            model_id=critic.model_id,
            usage=usage,
        )
        updated = replace(
            attempt,
            status=AttemptStatus.CRITICIZED if verdict is not None else AttemptStatus.CRITIC_FAILED,
            critic=outcome,
        )
        counters = state["counters"]
        return StateUpdate(
            attempts=_with_attempt(state["attempts"], updated),
            counters=replace(
                counters,
                critic_requests=counters.critic_requests + 1,
                provider_attempts=counters.provider_attempts + result.attempts,
            ),
        )

    def decide_revision(self, state: GenerationGraphState) -> StateUpdate:
        attempt = _active(state)
        used = state["revision_counts"].get(attempt.candidate_id, 0)
        decision = decide_revision(attempt, used, self._deps.config.revision)
        return StateUpdate(attempts=_with_attempt(state["attempts"], replace(attempt, revision_decision=decision)))

    def generate_revision(self, state: GenerationGraphState) -> StateUpdate:
        original = _active(state)
        decision = original.revision_decision
        assert decision is not None and decision.kind is RevisionDecisionKind.REVISE
        assert state["prompt"] is not None and state["guidance_image"] is not None
        request = state["request"]
        call = GenerationCall(
            candidate_id=original.candidate_id,
            candidate_index=original.candidate_index,
            attempt_index=original.attempt_index + 1,
            kind=AttemptKind.REVISION,
            prompt=build_revision_prompt(state["prompt"], decision.reasons, decision.instruction),
            seed=request.seed + original.candidate_index + _REVISION_SEED_OFFSET,
            source_page=request.canonical_page,
            guidance_image=state["guidance_image"],
            printable_region_mask=request.masks.printable_region_mask,
            previous_candidate=original.normalized.candidate_rgb if original.normalized else None,
        )
        record, counters = self._generate(call, state["counters"])
        record = replace(record, revision_of=original.attempt_id)
        counts = dict(state["revision_counts"])
        counts[original.candidate_id] = counts.get(original.candidate_id, 0) + 1
        return StateUpdate(
            attempts=(*state["attempts"], record),
            active_attempt_id=record.attempt_id,
            revision_counts=counts,
            counters=counters,
        )

    def rank_candidates(self, state: GenerationGraphState) -> StateUpdate:
        outcome = build_outcome(state["candidate_ids"], state["attempts"], self._deps.config.critic)
        return StateUpdate(outcome=outcome, finished_at=self._deps.clock.now())


_STATUS_ROUTES: Final[Mapping[AttemptStatus, str]] = {
    AttemptStatus.GENERATED: NORMALIZE,
    AttemptStatus.EXTRACTED: VALIDATE,
    AttemptStatus.VALIDATED: CRITICIZE,
    AttemptStatus.GENERATION_FAILED: DECIDE_REVISION,
    AttemptStatus.GEOMETRY_REJECTED: DECIDE_REVISION,
    AttemptStatus.VALIDATION_REJECTED: DECIDE_REVISION,
    AttemptStatus.CRITIC_FAILED: DECIDE_REVISION,
    AttemptStatus.CRITICIZED: DECIDE_REVISION,
}


def route_by_attempt_status(state: GenerationGraphState) -> str:
    """Send the active attempt to its next stage; every failure goes to the revision decision."""
    return _STATUS_ROUTES[_active(state).status]


def route_after_selection(state: GenerationGraphState) -> str:
    if state["active_attempt_id"] is None:
        return RANK
    return route_by_attempt_status(state)


def route_after_revision_decision(state: GenerationGraphState, max_revisions: int) -> str:
    attempt = _active(state)
    decision = attempt.revision_decision
    revisions_used = state["revision_counts"].get(attempt.candidate_id, 0)
    if decision is not None and decision.kind is RevisionDecisionKind.REVISE and revisions_used < max_revisions:
        return GENERATE_REVISION
    return SELECT_NEXT


def build_generation_graph(deps: GraphDependencies) -> CompiledStateGraph:
    """Compile the workflow with injected services; no client is created here."""
    nodes = _Nodes(deps)
    max_revisions = deps.config.revision.max_revisions_per_candidate
    graph = StateGraph(GenerationGraphState)
    graph.add_node(PREPARE, nodes.prepare_run)
    graph.add_node(GENERATE_INITIAL, nodes.generate_initial_candidates)
    graph.add_node(SELECT_NEXT, nodes.select_next_attempt)
    graph.add_node(NORMALIZE, nodes.normalize_candidate)
    graph.add_node(VALIDATE, nodes.validate_candidate)
    graph.add_node(CRITICIZE, nodes.criticize_candidate)
    graph.add_node(DECIDE_REVISION, nodes.decide_revision)
    graph.add_node(GENERATE_REVISION, nodes.generate_revision)
    graph.add_node(RANK, nodes.rank_candidates)

    stage_targets = [NORMALIZE, VALIDATE, CRITICIZE, DECIDE_REVISION]
    graph.add_edge(START, PREPARE)
    graph.add_edge(PREPARE, GENERATE_INITIAL)
    graph.add_edge(GENERATE_INITIAL, SELECT_NEXT)
    graph.add_conditional_edges(SELECT_NEXT, route_after_selection, [*stage_targets, RANK])
    graph.add_conditional_edges(NORMALIZE, route_by_attempt_status, [VALIDATE, DECIDE_REVISION])
    graph.add_conditional_edges(VALIDATE, route_by_attempt_status, [CRITICIZE, DECIDE_REVISION])
    graph.add_edge(CRITICIZE, DECIDE_REVISION)
    graph.add_conditional_edges(
        DECIDE_REVISION,
        lambda state: route_after_revision_decision(state, max_revisions),
        [GENERATE_REVISION, SELECT_NEXT],
    )
    graph.add_conditional_edges(GENERATE_REVISION, route_by_attempt_status, [NORMALIZE, DECIDE_REVISION])
    graph.add_edge(RANK, END)
    return graph.compile()


def run_generation(request: GenerationRequest, deps: GraphDependencies) -> GenerationRunResult:
    """Run the bounded workflow once. Nothing is approved or printed here."""
    graph = build_generation_graph(deps)
    final = graph.invoke(initial_state(request), config={"recursion_limit": deps.config.recursion_limit})
    outcome = final["outcome"]
    if outcome is None or final["prompt"] is None or final["guidance_image"] is None:
        raise RuntimeError("generation graph finished without a terminal outcome")
    assert final["started_at"] is not None and final["finished_at"] is not None
    return GenerationRunResult(
        run_id=final["run_id"],
        request=request,
        prompt=final["prompt"],
        guidance_image=final["guidance_image"],
        candidate_ids=final["candidate_ids"],
        attempts=final["attempts"],
        revision_counts=dict(final["revision_counts"]),
        counters=final["counters"],
        outcome=outcome,
        started_at=final["started_at"],
        finished_at=final["finished_at"],
        generator_model_id=deps.generator.model_id,
        critic_model_id=deps.critic.model_id,
        simulated_validation=deps.validator.simulated,
        config=deps.config,
    )
