# Track B: continuation generation and bounded visual criticism

This track turns a canonical A4 page, Track A's masks, and an optional prompt
into at most two ranked continuation candidates. Each candidate has a sanitized
new-ink mask and a preview. It never approves, prints, or exports anything:
every outcome has `requires_user_approval = True`.

## Public interface

```python
from printgpt_generation import (
    GenerationConfig, GraphDependencies, create_generation_request,
    run_generation, write_run_record,
)
from printgpt_generation.providers import SystemClock, UuidIdProvider
from printgpt_generation.track_a import TrackAValidator  # or SimulatedValidator

config = GenerationConfig()  # default layout: A4 at 300 DPI, 2 candidates, 1 revision
request = create_generation_request(canonical_page_rgb, mask_bundle, user_prompt, config=config, seed=0)
deps = GraphDependencies(
    generator=generator,      # CandidateGenerator
    validator=TrackAValidator(),
    critic=critic,            # VisualCritic
    clock=SystemClock(),
    ids=UuidIdProvider(),
    config=config,
)
result = run_generation(request, deps)          # GenerationRunResult
write_run_record(result, Path("artifacts/runs") / result.run_id)
```

- `canonical_page_rgb`: RGB `uint8`, shape `layout.image_shape + (3,)`
  (`(3508, 2480, 3)` by default). Convert alignment output from BGR exactly once,
  before this call.
- `mask_bundle`: Track A's `MaskBundle`. Track B reads `existing_ink_mask`,
  `protected_mask`, `allowed_region_mask`, `printable_region_mask`, `geometry`,
  and `config` (0/255 `uint8`, 255 = membership). An empty allowed region raises
  `ValueError`.
- `result.outcome.status` is `candidates_ready` or `no_valid_candidate`.
  `outcome.ranked` lists eligible candidates best first. `outcome.selections`
  explains the attempt kept for each candidate, including failures.
- The print overlay source for a selected attempt is
  `result.attempt(attempt_id).sanitized_new_ink_mask`. The raw provider image
  (`generation.output.image`) and the normalized `candidate_rgb` are diagnostics only.

`build_generation_graph(deps)` returns the compiled LangGraph graph for callers
who want to stream node updates. `initial_state(request)` builds its input.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `models.py` | Frozen domain records and typed status enums |
| `prompts.py` | Default/automatic prompt, revision and critic prompts |
| `providers.py` | Generator/critic protocols, provider errors, bounded retries, clock/ID sources |
| `doubles.py` | Deterministic generator and critic doubles |
| `guidance.py` | Region-guidance image (never printable) |
| `extraction.py` | Output normalization, registration checks, new-ink extraction |
| `critic.py` | Pydantic critic schema, parsing, threshold interpretation |
| `ranking.py` | Revision decision, eligibility, per-candidate selection, ranking |
| `track_a.py` | Structural protocols for Track A types, Track A adapter, simulated doubles |
| `requests.py` | Request validation and run-result record |
| `config.py` | `GenerationConfig` and the Pydantic `GenerationSettings` file format |
| `graph.py` | LangGraph state, nodes, routes, factory, `run_generation` |
| `metadata.py` | JSON-safe run record and artifact writer |
| `gemini.py` | Optional live Gemini adapters |
| `example.py` | Offline runnable example |

## Workflow

```text
prepare_run -> generate_initial_candidates -> select_next_attempt
  -> normalize_candidate -> validate_candidate -> criticize_candidate -> decide_revision
  -> (generate_revision -> normalize -> validate -> criticize -> decide_revision)
  -> select_next_attempt -> ... -> rank_candidates -> END
```

- Two initial candidates are generated with seeds `seed` and `seed + 1`.
  They are then processed one at a time from a queue.
- Failures route directly to `decide_revision`: geometry rejection, validation
  rejection, provider failure, and critic failure. The critic runs only on
  validated candidates, and it sees the original page, the region guidance, and
  the **sanitized preview**.
- Revision rules: a validation rejection revises with the validator's reasons. A
  geometry rejection revises with "keep framing". A critic verdict other than
  acceptable revises with the critic's problems and instruction. Provider and
  critic failures are not revisable (there is no semantic feedback). A revision
  never gets a second revision (`revision_counts` in state, re-checked by the
  route).
- Selection keeps the revision only if it is eligible and scores at least as
  well as an eligible original. Otherwise the original is kept.
- Eligibility means accepted validation AND a completed critic verdict that is
  not `reject` and meets every threshold.
- Ranking order: critic score, then mean sub-score, then lower `removed_fraction`,
  then candidate order. These are deterministic tie-breaks.
- Bounds: at most 4 generation requests and 4 critic requests. Transport retries
  are bounded per call (`RetryPolicy.max_attempts_per_call`, default 2) and per
  run (`max_total_attempts`, default 16), and never count as revisions. The
  worst-case path is 24 node steps; `recursion_limit` (default 60) is only a
  last-resort guard.

## Supported provider output and extraction

Provider output convention: the complete page, portrait, with the same framing
(no crop, border, rotation, or padding). Any resolution is accepted if its
aspect ratio is within 1% of the canonical raster. It is mapped with one
recorded uniform scale (`raw_to_canonical`). Other geometry is rejected as
`geometry_rejected`.

Registration: phase correlation of original ink against candidate ink estimates
global movement (maximum 0.5 mm). At least 85% of original ink must still be
present within 1.0 mm. Pages with fewer than 50 ink pixels record registration
as unverified.

New ink is extracted by registered-image comparison, not raw subtraction:

- Only dark (< 128) and nearly neutral (chroma ≤ 60) pixels count.
- Pixels within 1.0 mm of original ink are discarded, so redrawn or thickened
  originals are never reprinted.
- Components smaller than 0.5 mm² are discarded.

If a provider supplies an explicit additions layer, it is used instead and the
same filters apply. Guidance tints and the magenta outline cannot pass the ink
test.

Consequence: the discard band around original ink (1.0 mm) is wider than Track
A's default 0.5 mm clearance. New strokes therefore stop about 1 mm short of
original endpoints. The critic sees this gap in the preview. Narrow
`source_redraw_tolerance_mm` only together with `max_registration_shift_mm`;
the config enforces shift ≤ tolerance.

## Configuration

`GenerationConfig` holds `layout`, `candidate_count` (1–2), `revision`
(`RevisionPolicy`), `retry` (`RetryPolicy`), `extraction` (`ExtractionConfig`),
`critic` (`CriticThresholds`), and `recursion_limit`. Invalid values raise
`ValueError`.

External configuration can be loaded with
`GenerationSettings.from_json_file(path).to_config(layout)`. That model is
Pydantic and rejects unknown fields.

Critic thresholds default to an overall score ≥ 6 and each sub-score ≥ 5. They
are heuristic judgments, not calibrated probabilities.

## Run record

`write_run_record` writes `run.json` (UTF-8, two-space indent, final newline)
plus PNGs:

- `source_page.png` and `region_guidance.png`
- per attempt: `raw_output`, `candidate_rgb`, `candidate_new_ink_mask`,
  `sanitized_new_ink_mask`, and `preview_rgb`

The JSON holds run/candidate/attempt IDs, the prompt used, an
automatic-prompt flag, model IDs, config, the normalization transform,
registration/extraction metrics, validation metrics and reasons, critic
verdicts, revision decisions, call counters, timestamps, latency, failures,
usage, and relative artifact paths. Usage and cost stay `null` unless the
provider reports them; cost is never estimated. Credentials are never reachable
from the result. `simulated_validation` is `true` whenever the simulated
validator was used; such output is not print-ready.

## Offline example and tests

```powershell
python -m pip install -e .
python -m pip install -r requirements-generation.txt
python -m printgpt_generation.example artifacts/generation_example
python -m pytest
```

The example uses Track A's `printgpt_masking` when it is installed and the
simulated validator otherwise. All tests use deterministic doubles or a mock
HTTP transport. They need no network, API key, or paid requests. Test files
are `tests/test_generation_*.py`. `test_generation_fixtures.py` holds shared
helpers only.

## Gemini adapter

`printgpt_generation.gemini` provides `GeminiImageGenerator` and
`GeminiVisualCritic`. They use `google-genai` 2.31 and its Interactions API
(`client.interactions.create`), checked against the official docs and the
installed SDK in October 2026.

- Default model IDs are `gemini-nano-banana-2.1` (image) and `gemini-3.8-flash`
  (critic). Both are configurable through `GeminiSettings`.
- `create_gemini_client()` reads `GEMINI_API_KEY` only when called. It disables
  the SDK's internal retries, which otherwise made 4 requests per call, so
  `RetryPolicy` remains the only retry bound.
- Gemini cannot output A4 aspect. The adapter pads inputs on the right to 3:4
  (padding tinted as forbidden in the guidance), requests 3:4, rescales, and
  crops. Registration checks still validate the result.
- The critic requests JSON constrained to `CriticReport`'s schema. The response
  is still parsed as untrusted text.

**Live status: implemented and tested only against a mock transport.** No
Gemini credentials were available, so no live request has been made. Run the
live smoke test with `PRINTGPT_LIVE_GEMINI=1` and `GEMINI_API_KEY` set. It
makes paid requests.

## Integration steps with Track A

1. Merge `feature/masking-validation` (Track A) and this branch. The files do
   not overlap.
2. Install `requirements-generation.txt`, or add `langgraph` and `pydantic` to
   `pyproject.toml` (see the requests below).
3. Build masks once with `printgpt_masking.build_masks(page_rgb, allowed, layout=..., config=...)`.
4. Use `TrackAValidator(ValidationConfig(...))` as `deps.validator` in place of
   `SimulatedValidator`.
5. Run `python -m pytest`. `test_generation_track_a_integration.py` stops
   skipping once `printgpt_masking` is importable. This was verified on a
   temporary combination of both branches: 126 tests passed (live Gemini test skipped).
6. Feed the user's approval step from `outcome.ranked` and the selected
   attempt's `sanitized_new_ink_mask`.

## Requested shared-file changes (for the integrating developer)

- `pyproject.toml`: add `langgraph>=1.2,<2` and `pydantic>=2.7,<3` to
  `dependencies`. Optionally add an extra `gemini = ["google-genai==2.31.0"]`.
- `README.md`: mention `python -m printgpt_generation.example` and link to this
  document.
- After Track A merges, the simulated doubles in `track_a.py` can remain for
  tests. `MaskBundleLike` and `ValidationResultLike` may be replaced by direct
  imports from `printgpt_contracts`.
- `.gitignore`: add `*.egg-info/`. `pip install -e .` creates
  `src/printgpt_alignment.egg-info/`, which is currently untracked.

## Known limitations

- Only black line art on light paper is supported. Coloured strokes count as
  guidance colour and are discarded.
- Registration detects a global shift only. Local warping is caught only through
  the retention threshold.
- New strokes stop about 1 mm (`source_redraw_tolerance_mm`) short of original
  ink.
- Synchronous provider timeouts are enforced by adapters (the Gemini HTTP
  timeout), not pre-emptively by the graph.
- Critic scores are uncalibrated. Thresholds need tuning with real outputs.
- Gemini behaviour (framing fidelity at 3:4, schema compliance) is unverified
  live.
