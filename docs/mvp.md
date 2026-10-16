# PrintGPT integrated MVP

## Dependable path

The MVP supports black line drawings on light A4 paper, with or without
calibration markers. A supported raster image or PDF page is loaded and
rectified by the alignment component. It prefers marker alignment, then uses
the image bounds for scanner exports or detects the outer paper boundary for
photographs. Scanner exports do not need A4 proportions or visible background.
Non-A4 scans are fitted uniformly to the A4 canvas, with a warning about unknown
physical size; added padding is excluded from the allowed completion region.
The input-type control can explicitly select scan or photo handling. Low-contrast,
low-resolution, or blurry scans request a rescan. Photographs with unclear or
clipped page boundaries do too. A cropped scanner export cannot reveal what
content is missing, so users must check completeness. The OpenCV result is converted from BGR
to RGB exactly once. The user must define a rectangular allowed completion
region; the rest of the page is never implicitly allowed.

Track A detects existing ink, expands its protection clearance, and intersects
the allowed area with printer margins and layout exclusions. The LangGraph
workflow then asks the deterministic offline generator for two candidates.
Every attempt is normalized, checked for retained source geometry, converted
to a proposed new-ink mask, deterministically validated, and only then shown to
the deterministic critic as a sanitized preview. Each candidate may receive
one bounded revision. Ranking can select only attempts that passed validation
and criticism.

The integrated implementation is in `printgpt_mvp.pipeline`. It always uses
`TrackAValidator`; simulated validation is rejected at export. Generation
records and diagnostic images are saved under the run directory. A small
`mvp_manifest.json` records the selected attempt, validation result, approval
state, and export paths. `alignment.json` records the input mode, chosen method,
transform, quality measurements, and physical-size warnings.

The alignment notebook's synthetic and SmartDoc benchmarks intentionally use
`detect_and_rectify`, the strict marker detector. Their setup overlays a
calibration layout onto the source documents; they do not measure raw unmarked
scan acceptance. App uploads and the CLI use `detect_and_rectify_document`,
which retains the same marker implementation and adds the two unmarked paths.

## Integration failures found and fixed

The first merged-suite run could not collect tests because the generation
dependencies were not declared in `pyproject.toml`. LangGraph and Pydantic are
now core dependencies, while Streamlit and Gemini remain explicit extras.

The first true alignment-to-generation run then rejected every unchanged
candidate at registration. Track A's existing-ink mask intentionally includes
faint and antialiased pixels up to a much lighter threshold than generation's
dark-neutral candidate classifier. Comparing those unlike masks reported only
about 79% source retention even though the deterministic provider copied the
source exactly. Registration now compares the same dark-neutral class on the
canonical source and candidate. Track A's broader mask is still used without
weakening for redraw exclusion, protection, deterministic validation, and
export.

## Approval and export

Before approval, no `print/` output is created. Approval accepts only an
attempt present in the graph's ranked eligible results. The exporter checks
again that validation is real and accepted and that the sanitized mask is a
0/255 `uint8` array with exactly the canonical dimensions.

The PNG and PDF are built from `sanitized_new_ink_mask` only:

- mask value 255 becomes black toner;
- every other pixel is white;
- the source scan, raw model output, and complete-page candidate are never
  included in either printable file;
- the PDF MediaBox and placed raster are exactly 210 × 297 mm.

Print `approved_a4_actual_size.pdf` at **100% or Actual Size**. Disable Fit,
Shrink, Scale to Fit, borderless expansion, and automatic rotation.

## Commands

```powershell
python -m pip install -e ".[app]"
python -m pytest
printgpt-mvp-demo artifacts/mvp_example
python -m streamlit run src/printgpt_mvp/streamlit_app.py
```

The example creates a marked synthetic input, performs alignment, masking,
two-candidate generation, deterministic validation and criticism, selects the
highest-ranked eligible candidate, explicitly approves it in code, and writes
the toner-only PNG and exact-size PDF. Automated tests never call an external
model or printer.

## Known limitations and real-printer measurements

- The offline generator and critic demonstrate integration, not semantic model
  quality. The live Gemini adapter remains a separate verification task and a
  failed live provider must not prevent the offline demonstration.
- Existing-ink detection is tuned for dark line art on light paper. Shadows,
  coloured paper, halftones, and very faint pencil may require threshold tuning.
- Digital clearance cannot compensate for printer feed or duplex registration
  error. New strokes intentionally stop near protected source ink.
- The Canon imageRUNNER 2425i still needs repeated physical measurements of
  horizontal and vertical offset, rotation/skew, feed-direction variation,
  scale, non-printable margins, and run-to-run repeatability. Those results
  should determine a separate export correction and the safe clearance. They
  must not mutate the validated canonical mask or weaken deterministic checks.
