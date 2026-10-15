# PrintGPT: shared context and parallel-development contract

Read this file before either task brief. These documents are handoffs for two
developers and their AI agents working concurrently on the same project.

## Purpose and deadline

PrintGPT scans an A4 sheet containing an unfinished drawing, proposes a natural
continuation, and prints only the new marks onto that same physical sheet.
The user may supply a prompt or use automatic continuation with a default prompt.
The MVP offers two candidates, a preview of the actual new ink, and explicit
user approval before printing.

Hard deadline: Saturday, October 17, 2026, 23:59 Asia/Yerevan. No work after that
deadline is part of the delivery plan. Start with black line art on light paper.

## Repository and current state

Repository: https://github.com/saroyan-sw/PrintGPT

Use your own clone and a separate branch for each track. All paths below are
relative to the repository root; no particular Windows username or directory
is required. A local `.venv` is not included in Git. Create your own Python 3.12+
environment and install the repository with `python -m pip install -e .`.
Install pytest for tests. Consult `pyproject.toml` for declared dependencies;
do not assume packages on the original developer's machine exist on yours.

The A4 alignment component is implemented in `src/printgpt_alignment/`.
Read its README, geometry definitions, detection API, and tests before integration.
The user has reported successful printing and scanning with a Canon imageRUNNER
2425i; scanning is slow and physical integration will be revisited separately.
Alignment in image space does not yet establish repeatable printer-feed accuracy.

## Design decisions retained from the project discussions

- Manual scan upload is sufficient while scanner integration is unfinished.
- Automatic continuation with an empty user prompt is required.
- AI output cannot establish millimetre accuracy or preserve original pixels.
  Deterministic geometry and masks control the printable result.
- The source scan is immutable. Compose previews from it plus sanitized new ink.
- The visual critic sees the fitted, sanitized preview, including gaps introduced
  by protection. It cannot override deterministic validation.
- Allow one revision per candidate, then revalidate and re-criticize that result.
- No custom model training is required. Document datasets are evaluation material.
- Post-print rescanning is a separate calibration/evaluation workflow, not a
  required step for every normal generation. It is not model training.
- Scanner rectification happens before generation. A measured printer-output
  correction, when available, belongs at export; it is a separate transform.
- Final approval and exact-size PDF output are later integration work. Physical
  protection is limited by ink detection and actual printer registration error.

## Contract v1

Track A owns the shared Python models in `src/printgpt_contracts/`. Track B works
against the definitions below with local test doubles until those models arrive.
Do not create a second conflicting shared-model package.

Reuse `DEFAULT_LAYOUT`, `mm_to_px`, and `px_to_mm` from
`printgpt_alignment.geometry`. Default portrait A4 is 210 x 297 mm at nominal
300 DPI, with a rounded raster of 2480 x 3508 pixels. Origin is the top-left;
x increases right and y increases down. Arrays use row/column order.

All new public interfaces use RGB uint8 images of shape `(3508, 2480, 3)`.
The existing alignment code accepts OpenCV BGR/BGRA or grayscale and preserves
that representation when warping. Convert explicitly at the boundary to RGB;
do not silently reinterpret BGR as RGB or change the existing alignment API.

All binary masks are uint8, shape `(3508, 2480)`, values exactly 0 or 255.
255 means membership in the named mask, NOT a white printable pixel.

| Artifact | Meaning |
| --- | --- |
| `canonical_page` | Immutable aligned source image, RGB |
| `existing_ink_mask` | Detected original marks |
| `protected_mask` | Original marks expanded by the configured clearance |
| `allowed_region_mask` | User/task-defined completion area |
| `printable_region_mask` | Allowed area after protection, margins, and layout exclusions |
| `candidate_rgb` | Proposed complete-page image, for diagnostics only |
| `candidate_new_ink_mask` | Proposed additions extracted by Track B |
| `sanitized_new_ink_mask` | Track A's clipped additions |
| `preview_rgb` | Source with black ink applied where the sanitized mask is 255 |

The printable region is the intersection of the allowed region and safe page
interior, minus protected ink and layout exclusions. Exclude marker rectangles
AND their quiet zones. Reserve printed calibration labels/guides too; a central
completion rectangle is an acceptable initial choice. Missing allowed-region
information is an error, not permission to use the whole page.

Proposed shared API (names are contractual; implementation is pending):

```python
build_masks(canonical_page, allowed_region_mask, *, layout, config) -> MaskBundle
validate_candidate(candidate_new_ink_mask, masks, *, config) -> ValidationResult
compose_preview(canonical_page, sanitized_new_ink_mask) -> RGBArray
```

`MaskBundle` contains the five input/derived masks above and geometry/config
metadata. `ValidationResult` contains `accepted`, `reasons` (list of strings),
`metrics` (JSON-safe values), and `sanitized_new_ink_mask`. Accepted means the
sanitized result passes the configured checks; record violations in the original
proposal separately. Heavily clipped or nearly empty results must not pass just
because their remaining pixels are legal. Thresholds are configurable.

An all-white print raster corresponds to an empty mask; render ink as black
where mask equals 255. Never send the source scan itself as the print overlay.

## Safety clearance versus line continuity

Expanding existing ink creates a gap near stroke endpoints. Do not promise both
perfect joins and a positive exclusion gap. Expose the clearance in millimetres,
show its visual effect, and test endpoint cases. The critic may request a better
continuation but must never silently shrink protection or bypass it. Choosing
a real operating clearance requires physical registration measurements later.

## File ownership and integration

- Track A: `src/printgpt_contracts/`, `src/printgpt_masking/`,
  `tests/test_contracts.py`, `tests/test_masking*.py`, `docs/masking.md`.
- Track B: `src/printgpt_generation/`, `tests/test_generation*.py`,
  `docs/generation.md`, and `requirements-generation.txt` if needed.
- Preserve `src/printgpt_alignment/` and its tests. Avoid concurrent changes to
  the root README, `pyproject.toml`, and these handoff documents. Record requested
  shared changes in your own handoff notes for the integrating developer.
- Keep credentials in ignored local environment files; never commit keys,
  virtual environments, generated runs, or licensed dataset images.
- Each branch must include offline tests, one reproducible example, and notes
  listing public functions, dependency needs, limitations, and integration steps.

Both tracks can begin immediately with synthetic pages and mock masks. Integrate
the shared models first, replace Track B's validator test double with Track A's
implementation, then run generation -> validation -> criticism -> bounded
revision -> validation -> criticism -> selection. Add user approval and PDF
export afterward, with time reserved before the deadline for a real print test.

Task briefs: [Track A](TASK_A_MASKING.md) and [Track B](TASK_B_GENERATION.md).
