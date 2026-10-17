# PrintGPT integrated MVP

## Physical registration update

Preview acceptance no longer authorizes export onto the original sheet. The app
and CLI now retain PDF page boxes/rotation/UserUnit and raster scanner DPI, and
support confirmed whole-sheet, paper-corner, metadata and measured-crop geometry.
An arbitrary fitted scan remains usable for previews but cannot be exported until
its physical size and origin are established. PDF CropBox offsets and missing
coverage are preserved; unknown regions are excluded from generation.

See [physical alignment and Canon 2425i procedure](PHYSICAL_ALIGNMENT.md) for the
coordinate contract, assumptions, new notebook, 30-case marker-free software
benchmark, calibration target and independent repeated physical measurements.
Successful software tests are not measured printer accuracy. The optional printer
affine correction is separate from source registration and applies only at export.

## Dependable path

The MVP supports black line drawings on light A4 paper through two explicit
input modes. It never falls back from one mode to the other.

### Scanner A4

This mode is for a flat, full-page A4 image or PDF from a scanner. The shared
document loader renders PDFs at 300 DPI and records PDF page size and rotation
metadata. A configurable tolerance checks the physical dimensions when present,
the A4 aspect ratio, and the expected 300 DPI raster dimensions. Auto keeps a
portrait scan unchanged and applies the Canon imageRUNNER profile's 90°
counter-clockwise correction to landscape A4. The UI also offers left 90°,
right 90°, 180°, and no-rotation overrides.

After rotation, a correctly oriented page is normalized to exactly 2480 × 3508.
Known one-pixel renderer differences are resized deterministically and recorded,
including source/rotated/destination dimensions, scale factors, interpolation,
the complete 3 × 3 transform, padding, and the fact that no crop occurred. A
deliberately landscape override is fitted with white padding instead of being
stretched or cropped. Non-A4 input is rejected and is not assigned calibrated
physical coordinates. ArUco detection is bypassed only because the user chose
Scanner A4.

#### Printing onto an emailed or uploaded Scanner A4 scan

A Scanner A4 page is preview-only by default: normalization proves the page is
A4 and upright, not where the sheet sat on the scanner glass. Two things unlock
printing onto the original:

1. In the app, tick **"This scan shows the complete A4 sheet"** (scanned on the
   glass at A4, nothing cut off). Without a measured scanner profile the run is
   recorded as `awaiting_scanner_measurement`: generation and preview work, and
   export stays blocked.
2. Measure the scanner placement once (`printgpt_mvp.scanner_profile`):

   ```powershell
   # Scan a sheet with clear marks (the printed calibration page works) with Scan and Send.
   .\.venv\Scripts\python.exe -m printgpt_mvp.scanner_profile locate <scan.pdf>
   # Measure the same marker corners on the paper with a ruler (mm from the top-left corner),
   # write a printgpt-scanner-measurement/1 record (>= 6 spread fit points, >= 3 holdout points,
   # measurement_uncertainty_mm), then:
   .\.venv\Scripts\python.exe -m printgpt_mvp.scanner_profile fit <record.json>
   ```

   The fit uses only the fit points, and the holdout points judge it. The profile
   is saved as `measured` (default `%LOCALAPPDATA%\PrintGPT\scanner_profile.json`,
   override `PRINTGPT_SCANNER_PROFILE`) only when holdout error plus measurement
   uncertainty is below 0.5 mm.

With both, the scan is resampled into physical paper coordinates, the run is
`registered`, and `measured_source_accuracy_mm` records the bound. Printer
accuracy is a separate measurement (`printgpt_alignment.printer_calibration`).

### Marked page

This mode requires the expected ArUco marker quartet and retains the existing
perspective rectification, confidence score, and canonical A4 transform. It is
appropriate for marked calibration sheets and perspective-distorted captures.
A marker failure is shown as a marker failure; it never silently enters the
scanner path.

Both modes show source dimensions, detected orientation, applied correction,
warnings, and the normalized portrait A4 page. The user must explicitly confirm
that preview before the allowed-region controls can start masking and generation.
The source array is never mutated. The OpenCV result is converted from BGR to
RGB exactly once.

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
state, export paths, and complete input-normalization metadata. `alignment.json`
records the same mode, source dimensions, selected rotation, confidence,
calibration status, warnings, and transform.

The alignment notebook's synthetic and SmartDoc benchmarks intentionally use
`detect_and_rectify`, the strict marker detector. Their setup overlays a
calibration layout onto the source documents; they do not measure raw unmarked
scan acceptance. App uploads and the CLI use `detect_and_rectify_document`,
which retains the lower-level marker implementation and its legacy exploratory
unmarked paths. The MVP web interface does not use those fallback paths.

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
It also rejects unregistered preview-only source geometry. Export without a
measured printer profile is explicitly unmeasured and intended for sacrificial
verification first, not a guarantee of accurate overlay on a valuable original.

The PNG and PDF are built from `sanitized_new_ink_mask` only:

- mask value 255 becomes black toner;
- every other pixel is white;
- the source scan, raw model output, and complete-page candidate are never
  included in either printable file;
- the PDF MediaBox and placed raster are exactly 210 × 297 mm.

Print `approved_a4_actual_size.pdf` at **100% or Actual Size**. Disable Fit,
Shrink, Scale to Fit, borderless expansion, and automatic rotation.

## Printer calibration (print–scan loop)

The sidebar's **Printer calibration** measures the whole loop PrintGPT relies
on (scan the sheet → compute ink → print back onto it) with no ruler
(`printgpt_mvp.loop_calibration`):

1. **Start a new calibration** and download the round's page: 24 ArUco
   markers spread over A4 (`DICT_5X5_100`), as an exact-size PDF.
2. Print it at **100% / Actual size** from the tray you use for originals, and
   scan the printed sheet with Scan and Send, placed on the glass as for drawings.
3. **Measure** the latest emailed scan (or upload it). Half the markers fit an
   affine map from target to scan position; the other half check it. The table
   shows each round's real printed error, offset, and scale. "Fit to page"
   (scale ≈ 0.95) and skew are reported as problems and do not update the
   correction.
4. The next round's page is printed with the updated correction
   (`C' = C ∘ R⁻¹`, where `R` is the measured map). Repeat; the error should drop
   after the first correction and then settle at the printer's sheet-to-sheet
   variation.
5. **Save calibration** is enabled once a *corrected* round measures under
   0.5 mm on every marker. The profile is saved per printer queue and tray in
   `%LOCALAPPDATA%\PrintGPT\calibration\` (override `PRINTGPT_CALIBRATION_DIR`),
   so a demo reuses it without calibrating again.

With a saved profile, a Scanner A4 page confirmed as the complete sheet is
`registered` (evidence: the loop calibration), and every export is corrected
by the profile. A profile is valid only for the loop it measured: the same
scanner workflow, printer, tray, and 100% printing. Recalibrate if any change.

## Choosing the allowed area

The allowed rectangle is drawn on the normalized A4 preview (magenta) before
generation. `region_helper` suggests an area (blue): the empty strip on the side
where the drawing looks cut off (most strokes end at that edge), from the
drawing's edge to a 6 mm printable margin. "Use suggested area" fills in the
numbers. Before generation the rectangle is checked, and warnings explain when:

- it covers more than 25% of the existing drawing (new ink cannot go on or
  next to existing lines);
- the drawing looks cut off on one side but the area covers less than half of
  the space there (the continuation would be drawn outside and rejected);
- it is very small, or it reaches into the 6 mm margin the printer cannot print.

With Live Gemini on, generation with warnings needs an explicit "Generate
anyway" tick, so paid requests are not spent on an area that is probably wrong.

## Printing onto the original (main flow)

After approval, the app shows **Print onto your original** (`print_step`).
The printer cannot tell which sheet is in the tray or which way up it lies, so
loading the original is guided and confirmed by the user:

1. **Load your original sheet.** Instructions come from a paper-loading profile
   (`print_readiness.PaperLoadingProfile`: tray, drawing side up or down, edge
   fed first), saved in `%LOCALAPPDATA%\PrintGPT\paper_loading.json` (override
   `PRINTGPT_PAPER_PROFILE`) and editable in the app. A new or changed profile
   is **unverified**: the app asks for a spare test sheet. After a print, the user
   can mark the direction verified once the new ink landed on the drawn side,
   upright.
2. **Check the printer.** Each check is shown pass or fail: queue found and
   allowlisted, reachable on the network, ready, paper path clear (no jam, empty
   tray, open door or intervention), no other jobs waiting, and physical printing
   enabled (`PRINTGPT_HW_ALLOW_PRINT=1`). Only then is the artifact validated and
   a print job prepared, with the same checks as the hardware MCP server.
3. **Confirm and print.** The print button unlocks only after an explicit
   confirmation of how the sheet is loaded (plus the spare-sheet confirmation
   while unverified). It prints one page, one copy.
4. **Check the result**, and optionally rescan to compare placement.

The optional hardware expander now only reports printer and scanner status.

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
- Scanner A4 normalization establishes a canonical digital A4 coordinate system,
  not physical printer registration. Feed offset, scale, skew, printable margins,
  and run-to-run variation still require a measured printer calibration profile.
- Existing-ink detection is tuned for dark line art on light paper. Shadows,
  coloured paper, halftones, and very faint pencil may require threshold tuning.
- Digital clearance cannot compensate for printer feed or duplex registration
  error. New strokes intentionally stop near protected source ink.
- The Canon imageRUNNER 2425i still needs repeated physical measurements of
  horizontal and vertical offset, rotation/skew, feed-direction variation,
  scale, non-printable margins, and run-to-run repeatability. Those results
  should determine a separate export correction and the safe clearance. They
  must not mutate the validated canonical mask or weaken deterministic checks.
