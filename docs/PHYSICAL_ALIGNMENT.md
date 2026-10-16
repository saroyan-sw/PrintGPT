# Physical registration and Canon imageRUNNER 2425i verification

## What changed

Ordinary images and PDFs remain usable without markers. A successful preview is
not automatically a registered overlay. Export now requires source geometry
established under recorded assumptions. Printer accuracy remains **unmeasured**
until separate physical measurements have been made. Nothing prints automatically.
The demonstration generator's diagonal stroke is unrelated to alignment.

The earlier 802 x 1000 image was uniformly fitted to A4, adding about 17.6 mm at
the top and bottom. That fit is now preview-only. Do not simply stretch every
such image: its proportions could indicate cropping, scanner borders, resizing,
or a non-A4 original. Choose the evidence that actually describes the capture.

| Input evidence | Registration method | What must be checked |
| --- | --- | --- |
| Known ArUco layout | Automatic | Markers really were printed at actual size on this sheet |
| Entire unskewed A4 sheet, no border | Complete sheet | All image edges are physical paper edges; portrait reading orientation |
| Complete sheet was stretched | Complete sheet + resize acknowledgement | Whole sheet is present; this is not a crop |
| Visible paper corners, borders/skew/photo | Corners | Physical TL, TR, BR, BL in normalized source pixels; paper is flat |
| PDF | Metadata | MediaBox is the original paper size; scan placement within it is faithful |
| Raster with scanner DPI | Metadata | DPI is trustworthy; scan is uncropped, unskewed and begins at paper origin |
| Cropped scan or known scanner borders | Crop | Independently verified X/Y DPI and position of image top-left relative to paper top-left; no skew |
| Missing origin/scale | Preview only | Obtain corners, scanner settings, or measured offsets; rescan only if these cannot be recovered |

In the app open **Physical page registration**. Run a preview first to inspect
source pixel rulers, dimensions and recorded transforms. Correct the geometry,
select the confirmation, and generate again. Export approval is a separate action.
Changing geometry clears old candidates and downloads. For rotated raster scans
use corners in physical reading order; do not stretch a landscape image to portrait.
The numeric corner editor is manual, not automatic subpixel edge localization.

PDF page selection is one-based in the UI and zero-based in Python. PDF rendering
uses PDFium; pypdf retains MediaBox, CropBox, inherited rotation and UserUnit.
Visible CropBox content stays at its original offset within MediaBox; it is not
stretched to fill A4. PDF display rotation is undone for physical registration.
A PDF can contain a cropped/rescaled scan inside an otherwise correct A4 page:
the container's dimensions alone cannot establish the embedded image's geometry.
Malformed/unverifiable metadata does not silently certify registration. Non-A4
physical pages can be previewed at their physical scale but cannot be exported
as registered portrait-A4 originals.

## Coordinate contract

1. Source pixels are decoded once. EXIF orientation is applied and X/Y DPI axes
   are swapped where necessary. Source coordinate rulers refer to these pixels.
2. Registration maps source pixels to canonical page coordinates using a recorded
   homography. Canonical outer edges span exactly 210 x 297 mm. At 300 nominal DPI
   this is 2480 x 3508 pixels. Exact per-axis scales are `W/210` and `H/297`.
   Nominal `mm_to_px(dpi)` remains for legacy raster sampling/marker layout;
   rounded raster dimensions cause a small distinction from exact edge mapping.
3. Missing source coverage, cropped PDF regions and letterbox padding are excluded
   from generation, with an additional ink-clearance erosion at coverage edges.
   Existing-ink and printer-margin masks still apply. Geometry does not restore
   ink missing from a blurred or cropped source.
4. Candidate previews and sanitized new-ink masks share that canonical space.
   A heuristic detection confidence is not a physical accuracy score.
5. Export draws only new ink over exactly 210 x 297 mm. PDF coordinates are
   `x_pt = x_mm * 72/25.4`, `y_pt = (297-y_mm) * 72/25.4`.
6. Optional printer correction is applied only at export: if measured mechanics
   map commanded millimetres to physical millimetres with affine `P`, command
   `inverse(P)` times the intended position. Source geometry and protection masks
   are not rewritten. Corrected PNG and PDF contain the same corrected raster.
   New ink that would be clipped or enter measured unprintable margins is rejected.

OpenCV samples at pixel centres whereas physical extents are outer edges. This
introduces subpixel sampling uncertainty, not a new physical origin. Independent
rendered-ink checks measure this too. Printer correction uses nearest-neighbour
resampling of binary toner; a pixel-diagonal allowance is included in measured
protection budgeting. Prefer the PDF, not a PNG viewer's automatic print scaling.

`alignment.json` preserves metadata, options, source/canonical/mm transforms,
registration evidence, warnings and optional independent controls. The run manifest
also records registration and printer evidence. A confirmed source without printer
calibration exports as **unmeasured** for sacrificial verification, not as a guarantee
of collision-free overlay on a valuable original.

## Python / command line

```python
from printgpt_alignment.registration import RegistrationOptions
from printgpt_mvp import canonicalize_document

page_rgb, evidence = canonicalize_document(
    "scan.pdf",
    registration=RegistrationOptions(mode="metadata", confirmed=True),
)
# Only confirm after checking the assumptions in the table.
```

Pass the same `registration` option to `run_offline_document`. The lower-level
`load_document_page` still returns pixels for backwards compatibility; use
`load_document` for metadata and `register_document` for physical evidence.
Low-level detector acceptance alone is not an export authorization.

```powershell
.venv\Scripts\python -m printgpt_alignment.cli rectify scan.pdf --output artifacts\rectified.png --report artifacts\alignment.json --registration geometry.json
```

Example `geometry.json`: `{"mode":"full_page","confirmed":true}`. Default without
this file remains preview-only unless the known ArUco layout is detected.

## Software checks (not printer measurements)

```powershell
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m printgpt_alignment.physical_benchmark --output artifacts\physical_alignment
```

If Windows denies the shared pytest temporary directory, add a **new, unused**
`--basetemp=artifacts\tests-run-UNIQUE` directory. Pytest clears its selected base.

The 30-case benchmark preserves true transforms and checks 25 independent dot
centres, not the corners used to fit geometry. It covers borders, scale, translation,
perspective, four orientations, blur/noise/shadow, and known crops. It adds small
corner-annotation noise and saves inputs, rectified images, expected-green versus
recovered-red overlays, corner errors, independent-point errors and failure rates.
Corners/offsets are supplied: this is **not a benchmark of automatic paper detection**.
The acceptance target is <0.5 mm for both recovered corners and independent dots.
Tests additionally render exported PDFs and inspect ink at independent locations,
exercise metadata conflicts, CropBox offsets, EXIF, non-A4 and blocked exports.
Real documents without surveyed landmarks can test usability, not absolute accuracy.

## Canon 2425i: physical measurement procedure

### 1. Fix and record the configuration

- Use flat A4 sheets from one batch, mark the physical top/feed edge in pencil,
  and record stock/weight, tray, simplex/duplex side, orientation, driver/version,
  application and every scaling/rotation setting. Start with simplex and one tray.
- Print PDF at **Actual Size / 100%**. Disable fit, shrink, borderless expansion,
  automatic rotation and any application/driver enlargement. Confirm A4 both in
  the PDF viewer and driver. A configured 300 DPI is not proof of physical size.
- Prefer the platen glass for calibration scans. Canon describes platen capture
  as more accurate than the moving feeder and specifies placement at the top-left
  glass corner. Use that as a reproducible starting point, not a zero-offset claim.
  See [Canon's original-placement instructions](https://oip.manual.canon/USRMA-4816-zz-CSL-2425-enUV/contents/devu-basicope-pset_original.html)
  and the [2425/2425i user guide](https://oip.manual.canon/USRMA-4811-zz-CSL-2425-enGB/index.html).
- Scan at a fixed 300 or 600 DPI with fixed A4 scan area. Disable auto-crop,
  deskew, page fitting and post-scan resizing where the selected workflow permits.
  UI names differ by driver/firmware; record actual settings rather than assuming.
  Never assume a model-specific guaranteed printable margin or registration tolerance.

### 2. Establish measurement geometry independently

Verify scanner scale and origin using an independently measured flat reference
(ruler/grid) and physical paper edges. Check both axes and centre/edge locations.
Record measurement uncertainty. Do not calibrate a printer from scans auto-fitted
to the printer's own marks: that can cancel the very scale/offset error being tested.
Do not use the same printed targets to both fit and validate scanner geometry.
For precision measurement, use suitable measuring equipment; a coarse ruler cannot
substantiate sub-0.5 mm performance. Paper curl, lens distortion and scanner nonlinear
distortion can invalidate a homography and must show up in independent checks.

### 3. Print and measure an uncorrected target

```powershell
.venv\Scripts\python -m printgpt_alignment.printer_calibration target --pdf output\pdf\canon2425i_registration.pdf --measurements artifacts\canon2425i\fit.json
```

The page has 35 labelled cross centres and uncorrected 1-12 mm edge ladders.
Coordinates are measured from the physical top-left paper corner. Printed labels
are rounded for readability; use exact `expected_mm` from JSON for calculations.
Print at least three uncorrected sacrificial sheets. Fill `measured_mm` in JSON,
duplicate point rows with distinct sheet IDs for repeats, use a unique session ID,
and complete the settings. Keep fit and holdout locations separate. Measure
cross centres, not stroke edges. Record the visibility/clipping of edge ladders
on all four sides. Choose safe margins larger than the worst measured clipping
boundary, allowing feed variation; the software's initial 10 mm margin is not a
Canon specification. Repeat separately for duplex sides or other trays/stocks.

```powershell
.venv\Scripts\python -m printgpt_alignment.printer_calibration fit artifacts\canon2425i\fit.json --output artifacts\canon2425i\provisional.json
```

This estimates X/Y scale, offsets, rotation and affine shear using only fit points.
Inspect independent holdout residuals. A systematic pattern/nonlinear residual
means this affine model is insufficient; investigate scanner calibration, paper
flatness and feed before using it. Do not hide the problem with another page fit.

### 4. Verify corrected prints and repeatability

```powershell
.venv\Scripts\python -m printgpt_alignment.printer_calibration target --profile artifacts\canon2425i\provisional.json --pdf artifacts\canon2425i\corrected.pdf --measurements artifacts\canon2425i\verification.json
```

Print **at least five separate corrected sheets**, including a reloaded stack and
a later run. For each, measure the same nine or more independent positions spread
across the page (prefer all 35). Do not fit another transform to those results.
Copy/extend the template points with unique sheet IDs, fill real `measured_mm`,
use a new session ID, record `measurement_uncertainty_mm`, set
`measurement_geometry_verified` only after step 2, and enter measured
`safe_margins_mm` in left/top/right/bottom order. Preserve the correction ID and
identical settings. Compare against the original **intended** coordinates.

```powershell
.venv\Scripts\python -m printgpt_alignment.printer_calibration validate artifacts\canon2425i\provisional.json artifacts\canon2425i\verification.json --output artifacts\canon2425i\measured_profile.json
```

The report includes mean/max/p95 error and maximum displacement between repeated
measurements at the same location. A profile is measured only for physical data,
verified measurement geometry, at least five sheets and nine common locations,
and worst error **plus measurement uncertainty <0.5 mm**. Synthetic records cannot
qualify. The export API recomputes the fit/validation instead of trusting a status
label. Enter this profile path in the app for export correction. A profile is valid
only for its recorded conditions; retire it after changes or drift.

### 5. Verify the complete second-pass overlay and protection budget

On a sacrificial sheet, add known reference marks, capture it, register it, then
print additions near at least nine surveyed independent locations. Check actual
second-pass displacement against the original marks, including after unloading
and reinserting sheets. This exercises scanner and paper-feed errors together.
Repeat five times; never start with an irreplaceable original.

Source checks can be recorded with `RegistrationOptions(control_points_px=...,
control_points_mm=..., measurement_uncertainty_mm=...)`. Use at least nine distinct
locations across the page, not the registration corners/markers. Failed independent
source checks block export. `evaluate_control_points` reports displacement in mm.

The default 0.5 mm ink clearance cannot safely absorb a 0.5 mm source error PLUS a
0.5 mm printer error. With both measurements available, export checks:

`source bound + printer bound + raster allowance < configured ink clearance`.

If this fails, increase clearance and regenerate the masks/candidates, or improve
the capture/calibration. Without source measurements this combined budget remains
explicitly unverified, even with a measured printer profile. Also retain a practical
reserve for future feed variation; observed maxima are not future guarantees.
Printer margins and ink clearance solve different problems and must both hold.

## Current limitation

Implementation verification on 2026-10-16:

- Full repository suite: 219 passed, 1 optional live-model test skipped.
- Final 30-case supplied-geometry benchmark: 30 accepted, zero failures against
  the 0.5 mm target; maximum independent dot-centre error 0.12012 mm. Results and
  overlays are under `artifacts/physical_alignment_final/`.
- All 15 existing FUNSD scans opened as previews, correctly remaining unregistered
  without additional physical evidence. This is a usability check, not metrology.
- The new notebook executed end to end; an executed copy is under
  `artifacts/physical_alignment/executed_notebook.ipynb`. The existing MVP notebook
  was not edited. The calibration PDF was rendered and visually inspected.

Software tests establish digital coordinate consistency. No Canon hardware was
operated during implementation. No physical scale/offset, printable margin,
repeatability or second-pass overlay accuracy has yet been measured here.
