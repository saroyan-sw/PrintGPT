# Track B: continuation generation and bounded visual criticism

Give this brief to the developer/AI agent responsible for generation.
First read [shared project context and contract](PARALLEL_DEVELOPMENT.md).
Implement this track in your own checkout/branch. Track A is simultaneously
building the shared models, ink masks, and validator. Start using test doubles
matching that interface; label their output as simulated, not print-ready.

## Work to implement

Implement a provider-independent workflow accepting the canonical RGB image,
Track A's MaskBundle, and an optional user prompt. Expose injectable generator,
validator, and critic interfaces so the whole flow runs offline in tests.

When the prompt is empty or whitespace, use this default:

> Continue the unfinished drawing naturally. Match its line thickness, style,
> perspective, and subject. Keep existing content unchanged. Add content only
> inside the supplied printable region. Do not add text, borders, backgrounds,
> shading, or unrelated objects. Respect excluded areas even if this leaves a
> small gap at an existing stroke endpoint.

Start with simple black line drawings on white/light paper. Supply the model
with the original and clearly distinguished region guidance. Guidance colours,
labels, or mask patterns must never become printable content.

Generate two initial candidates. Keep the source immutable and retain raw model
output separately. Normalize each proposal to canonical coordinates with an
explicit recorded transform before extracting proposed new ink. A model may
redraw, shift, crop, or change the aspect ratio of the source: blindly stretching
the entire generated image does not solve registration. Document the supported
output convention, check registration against stable source content where
possible, and reject unsupported geometry rather than claiming exact fidelity.

Track B owns extraction of `candidate_new_ink_mask` from provider output. Prefer
an explicit additions layer when supported; otherwise implement and test a
documented registered-image comparison that avoids reprinting original strokes.
Do not use naive raw pixel subtraction as proof that differences are new art.
Tests must cover a provider redrawing/shifting an original stroke and adding
background noise. Pass proposed additions to Track A for final enforcement.

## Bounded workflow

For each candidate:

1. Normalize/extract proposed ink and run deterministic validation.
2. If validation succeeds, compose the sanitized preview and run the critic on
   the original, allowed/printable-region guidance, and actual sanitized preview.
3. If the validator fails, use its reasons for a possible revision; a critic call
   is unnecessary for a structurally invalid candidate.
4. Permit at most one revision for that candidate. It must pass normalization,
   validation, and criticism again. Retain the original valid candidate if the
   revision is worse or fails.
5. Rank only candidates with successful validation and completed, acceptable
   criticism. Return both candidate records including failures. If none qualify,
   return an explicit no-valid-candidate outcome; do not select a failed result.

Critic output is structured and schema-validated: score, boundary-continuity
score, style-match score, and geometry/plausibility score (each 0-10), problems
(list of strings), revision instruction, and recommendation (accept/revise/reject).
The critic evaluates visible gaps, continuity, line width, plausible proportions,
relevance, and artifacts. It cannot relax protection. Use configurable quality
thresholds; scores are heuristic judgments, not calibrated probabilities.

With separate calls per candidate, the logical budget is at most four image
generations and four critic evaluations per run. Bound transport retries,
timeouts, and total call attempts separately so retries cannot create an
unbounded loop. Preserve partial results and errors. Final human approval is a
later integration step; never print from this workflow.

## Provider, orchestration, and metadata

Use LangGraph for the workflow's explicit bounded state transitions. Keep the
generation/critic adapters replaceable. Implement deterministic generator AND
critic doubles so tests need no API key, paid requests, or network. LangGraph is
required for orchestration in this track; model/provider logic must remain usable
and testable independently of the graph.

## Project code style and LangGraph structure

Match the style already used in `src/printgpt_alignment/`. Begin modules with a
short module docstring and `from __future__ import annotations`. Use complete
type hints, small functions with one clear job, descriptive names, explicit
validation, and immutable value objects where practical. Existing project data
objects use `@dataclass(frozen=True, slots=True)`; follow that pattern for
internal domain values. Use Pydantic at untrusted boundaries such as model JSON,
saved run metadata, and configuration loaded from outside Python.

Keep domain logic independent of orchestration. Suggested package structure:

```text
src/printgpt_generation/
    __init__.py       # small public API only
    models.py         # typed domain and provider results
    prompts.py        # default and revision prompt construction
    providers.py      # protocols/interfaces and deterministic doubles
    extraction.py     # candidate normalization and proposed-ink extraction
    critic.py         # critic schema and interpretation
    ranking.py        # deterministic eligibility and ranking rules
    graph.py          # LangGraph state, nodes, routes, and graph factory
    metadata.py       # JSON-safe run records
```

The filenames can change when there is a clear reason, but preserve these
boundaries. `graph.py` should coordinate tested services rather than contain
image processing, prompt construction, provider-specific parsing, or ranking
formulas.

Represent graph state with a typed schema whose fields have one documented
meaning. Include the immutable request, candidate records, current candidate
ID, attempt/revision counts, errors, and terminal outcome. Avoid a loose
`dict[str, Any]`. Return explicit partial state updates from nodes; do not mutate
shared lists, NumPy arrays, or nested records in place.

Each LangGraph node should perform one visible transition, such as generating
initial candidates, normalizing one candidate, validating one candidate,
criticizing one preview, deciding whether to revise, or finalizing ranking.
Name conditional routing functions after the decision they make and keep them
pure. Route using typed status values/enums rather than checking for the
presence of arbitrary dictionary keys.

Build the graph through a factory that receives generator, validator, critic,
clock/ID providers, and configuration as dependencies. Do not instantiate live
clients at import time or read environment variables inside graph nodes. This
makes the same compiled workflow runnable with deterministic doubles and live
adapters. Keep the initial MVP execution synchronous unless a provider forces
an async boundary; do not mix sync and async calls across otherwise identical
nodes.

The graph must be finite by construction. Encode the single-revision rule in
state and routing, define an explicit terminal result for success and failure,
and also set a conservative LangGraph recursion limit as a final guard. Provider
transport retries belong inside a bounded adapter policy and must not increment
the semantic revision count. Every generated revision re-enters normalization,
validation, and criticism instead of jumping directly to ranking.

If checkpointing is added, persist compact JSON metadata and artifact paths
rather than embedding full NumPy arrays or encoded images in checkpoints.
Checkpointing is optional for the MVP; a deterministic run record is required.
Do not make core execution depend on a database or external service.

Follow the existing repository's error style: raise `ValueError` for invalid
programmer/configuration inputs, use explicit result states for expected model
or candidate failures, and avoid broad exception swallowing. Use `pathlib.Path`,
UTF-8 JSON with two-space indentation and a final newline, deterministic seeds
for doubles, and no mutable module-level state. Add comments for reasons and
constraints rather than narrating obvious statements.

Tests should resemble the current focused pytest tests: arrange a small
deterministic input, call a public boundary, and assert observable behavior.
Name tests after the behavior. Test graph transitions and call counts through
fakes instead of asserting LangGraph internals. A mock run with identical seed
and input must produce identical decisions and metadata apart from injected
time and IDs.

For a live Gemini adapter, verify current official SDK/model capabilities before
coding. Keep model IDs configurable and credentials in ignored environment
variables/files. Report unavailable models or credentials clearly. Live smoke
tests and actual cost are separate from the offline success criteria.

Use typed models for candidate state, critic result, and revision history. Keep
provider-specific models inside this track; consume Track A's shared contract.
Persist small JSON metadata: run/candidate IDs, prompt used, model ID/settings,
artifact paths, normalization transform, validation metrics/config, critic
results, attempt/revision counts, timestamps, latency, and errors. Store images
and masks separately. Record usage/cost only when supplied or computable from
verified pricing; otherwise mark it unknown. Never serialize credentials.

## Tests and deliverables

Cover the empty-prompt mode, two candidates, mask conventions, geometry failures,
source preservation, validation rejection, one-revision limit, validation after
revision, malformed critic output, timeout/provider failure, a failed revision
after an accepted original, and no-valid-candidate outcomes. Check that the
critic receives the sanitized preview rather than the raw generated image.

Deliver `src/printgpt_generation/`, isolated tests, an offline runnable example,
and `docs/generation.md` describing imports, configuration, dependencies, API
boundaries, live-test status, and remaining limitations. Record added packages
in this track's `requirements-generation.txt`; let the integrator reconcile
shared dependency declarations. No UI, printing, or PDF-export implementation
is required for this track.
