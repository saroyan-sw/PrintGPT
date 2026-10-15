# Track A: existing-content masking and deterministic validation

Give this brief to the developer/AI agent responsible for content protection.
First read [shared project context and contract](PARALLEL_DEVELOPMENT.md).
Implement this track in your own checkout/branch. The other developer is building
generation and criticism concurrently; do not wait for their live model adapter.

## Work to implement

1. Inspect the existing geometry/layout and reuse its conversion functions.
2. Implement the shared models/API from contract v1 in `printgpt_contracts`.
   Keep that small package free of generation-provider dependencies.
3. Detect dark original ink on light backgrounds. Start with deterministic
   thresholding and configurable background/noise handling. Preserve faint and
   thin legitimate strokes; do not erase them merely to clean up the mask.
4. Expand original ink by configurable clearance in millimetres. Provide an
   explicit zero-clearance test mode; do not describe any default as physically
   calibrated. Use a documented Euclidean-distance convention for dilation.
5. Combine the caller's allowed region with protection, safe margins, marker
   quiet zones, and other layout exclusions. The current layout's 10 mm margin
   is a provisional design margin, not a measured printer specification.
6. Validate shape, dtype, binary values, and geometry at the interface. Reject
   malformed inputs instead of silently resizing masks.
7. Measure proposed violations before clipping. Intersect proposed ink with the
   printable region, then measure and evaluate the sanitized result.
8. Implement preview composition using an unchanged source plus black new ink.
   Expose diagnostics without changing the source image in place.

## Validation measurements

Report proposed and final ink counts; overlap with protected ink; marks outside
the allowed region; margin/layout violations; removed pixel count and fraction;
final connected-component count; small-component/noise fraction; and rejection
reasons. Report applicable areas in mm² using the layout's nominal DPI.
State the denominator for every fraction (page area, printable area, or proposed
ink). Handle zero denominators explicitly.

Configurable acceptance checks should reject empty/nearly empty output,
excessive coverage, excessive clipping, and excessive scattered noise. A few
clipped pixels may be repairable, while a proposal losing most of its content
must fail. Record the configured thresholds with the result. These checks do
not establish semantic quality; the critic owns that assessment.

The final sanitized mask must have zero intersection with all forbidden areas.
This is a digital invariant, not a guarantee that a real printer has zero
mechanical alignment error. Do not add original ink to bridge an endpoint gap.

## Tests and deliverables

Use known synthetic strokes and masks to check protection, physical dilation,
each exclusion type, malformed inputs, zero allowed area, clipping fractions,
empty/excessive output, and preservation of the source array. Include a case
where a valid continuation approaches a protected endpoint and exposes the gap.
Check RGB channel handling with a small coloured diagnostic input as well.

Provide diagnostic images for existing ink, protection, allowed/printable
regions, removed proposed pixels, sanitized ink, and the composed preview.
Provide a runnable offline example and `docs/masking.md` describing imports,
configuration, test command, measurements, limitations, and integration notes.
Keep large generated artifacts outside Git. Add no model calls or printer jobs.

Acceptance: Track B can call the documented interface using a synthetic
candidate and receive a reproducible validation decision, safe mask, and preview.
