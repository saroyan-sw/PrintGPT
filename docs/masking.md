# Track A: masking and deterministic validation

This track protects existing page content and clips proposed additions to an
explicit printable region. It does not call a model, judge semantic quality,
print, or claim physical registration accuracy.

## Public interface

```python
from printgpt_alignment import DEFAULT_LAYOUT
from printgpt_contracts import MaskingConfig, ValidationConfig
from printgpt_masking import build_masks, compose_preview, validate_candidate

masks = build_masks(
    canonical_page_rgb,
    allowed_region_mask,
    layout=DEFAULT_LAYOUT,
    config=MaskingConfig(clearance_mm=0.5),
)
validation = validate_candidate(
    candidate_new_ink_mask,
    masks,
    config=ValidationConfig(),
)
preview_rgb = compose_preview(
    canonical_page_rgb,
    validation.sanitized_new_ink_mask,
)
```

`canonical_page_rgb` is RGB `uint8` in row/column order. At the default layout
it has shape `(3508, 2480, 3)`. OpenCV alignment output is BGR, so Track B must
convert it explicitly at the boundary:

```python
canonical_page_rgb = cv2.cvtColor(rectification.warped, cv2.COLOR_BGR2RGB)
```

All masks are `uint8`, have the layout's `(rows, columns)` shape, contain only
`0` or `255`, and use `255` to mean membership. Inputs are validated and never
resized. An allowed-region mask is mandatory; `None` is an error.

The public models are:

- `MaskingConfig`: ink thresholds, background/noise handling, physical
  clearance, printer margin, and layout-exclusion controls.
- `ValidationConfig`: minimum addition size, maximum coverage/clipping/noise,
  optional component limit, and optional rejection of every forbidden contact.
- `MaskBundle`: existing ink, protected ink, allowed region, safe page interior,
  layout exclusions, printable region, and JSON-safe geometry/config metadata.
- `ValidationResult`: `accepted`, `reasons`, JSON-safe `metrics`, and the
  `sanitized_new_ink_mask`.

`detect_existing_ink` and the diagnostic helpers are also public. No additional
dependency is required beyond NumPy and OpenCV already declared by the project.

## Detection and geometry

Ink detection combines an absolute grayscale threshold with local background
contrast. The latter preserves faint strokes under smooth paper/shadow
variation. Optional median filtering and connected-component cleanup are off by
default because cleanup can erase real thin marks.

Protection uses OpenCV's precise Euclidean distance transform. A background
pixel belongs to the protected mask when its centre is at most
`mm_to_px(clearance_mm, layout.dpi)` pixels from detected ink. The default
clearance is provisional, not physically calibrated. Zero protection is only
available through `zero_clearance_test_mode=True`; it is intended for tests.

The printable mask is exactly:

```text
allowed region
AND safe page interior
AND NOT protected existing ink
AND NOT layout exclusions
```

Layout exclusions cover all four ArUco marker rectangles and quiet zones, the
calibration text/marker labels rendered by `calibration_page.py`, and a
configurable band around the printed protected-content guide. Printer margin
and every clearance are specified in millimetres.

Positive clearance necessarily creates a visible gap where a continuation
approaches an existing stroke endpoint. Diagnostics expose that gap; the
validator never copies original ink, bridges it, or silently reduces clearance.

## Validation metrics and denominators

The proposal is measured before clipping. The sanitized mask is its exact
intersection with the printable region and is checked to have zero digital
intersection with every forbidden pixel.

- `removed_fraction`: removed unique pixels / proposed ink pixels. It is `0.0`
  for an empty proposal.
- `proposed_coverage`: proposed ink pixels / all page pixels.
- `final_coverage`: sanitized ink pixels / printable-region pixels. It is `0.0`
  when the printable region is empty.
- `small_component_fraction`: sanitized pixels in components at or below the
  configured small-component size / sanitized ink pixels. It is `0.0` for an
  empty sanitized mask.

The report also includes proposed/final counts, protected overlap, pixels
outside the allowed region, margin and layout violations, connected-component
count, pixel areas, corresponding nominal mm² areas, masking configuration, and
validation thresholds. Overlap counters are independent and may refer to the
same proposed pixel; `removed_pixel_count` is the unique total.

Minor clipping may be accepted when it stays below `max_removed_fraction`.
Nearly empty, over-covered, heavily clipped, or excessively scattered results
are rejected. Set `reject_any_forbidden_intersection=True` where the caller's
policy requires rejection on even one clipped pixel.

## Offline example and tests

From an editable installation:

```powershell
python -m printgpt_masking.example artifacts/masking_example
python -m pytest
```

The example writes `existing_ink.png`, `protected_ink.png`,
`allowed_region.png`, `printable_region.png`, `removed_candidate_pixels.png`,
`sanitized_new_ink.png`, `final_preview.png`, `decision_overlay.png`, and a
JSON validation report. Generated artifacts remain outside Git.

## Track B integration

1. Convert the accepted alignment raster from BGR to RGB exactly once.
2. Obtain an explicit task/user allowed-region mask; do not substitute an
   all-page mask when that information is missing.
3. Call `build_masks` once for the immutable source page and reuse its
   `MaskBundle` for both candidate attempts.
4. Have generation return `candidate_new_ink_mask` in canonical coordinates;
   a complete-page candidate RGB image is diagnostic only.
5. Call `validate_candidate` before visual criticism. Give the critic the
   sanitized `compose_preview` result and validation metrics, including the
   clearance gap. The critic may request one revision but cannot override masks.
6. Revalidate every revision and select only an accepted result. Keep
   `sanitized_new_ink_mask` as the sole later print-overlay source; an empty
   mask corresponds to an all-white print raster.
7. Add user approval and exact-size export after candidate selection. Any future
   measured printer correction belongs at export and must not mutate this mask.

## Known limitations and shared-file requests

Dark-on-light line art is the supported mode. Thresholds may need tuning for
coloured paper, severe shadows, halftones, or very faint pencil. Calibration
label rectangles follow the current V1 renderer and must be updated alongside a
future layout version. Nominal digital clearance cannot guarantee physical
separation because printer feed error is not yet measured.

No shared-file change is requested. Package discovery already includes the new
modules, and all runtime dependencies are already declared in `pyproject.toml`.
