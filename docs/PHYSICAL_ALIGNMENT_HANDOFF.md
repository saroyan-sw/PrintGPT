# PrintGPT: reliable alignment when printing onto the original sheet

## Task for a separate chat

Continue work on https://github.com/saroyan-sw/PrintGPT, starting from the latest
`feature/mvp-integration` branch. The main local checkout is
`C:\Users\saroy\Codes\PrintGPT`; `C:\Users\saroy\Codes\PrintGPT-app` is a separate
app worktree. Inspect status and preserve existing notebook changes before editing.

Implement and verify reliable physical alignment for printing generated additions
onto an existing document, including ordinary A4 scans without ArUco markers.
Distinguish a preview that fits on an A4 canvas from a physically registered overlay
that will land in the intended position on the original paper.

## Reproduction and confirmed cause

An uploaded form is 802 x 1000 pixels. The app now accepts it as a scanner export,
preserves its proportions, fits it to the canonical A4 raster, and centers it.
At 300 DPI the transform scales by about 3.09227 and adds about 207.87 pixels
(17.6 mm) above and below. This does not establish the original paper size or
the position of its edges. Printing that overlay onto the original sheet may
misalign the new ink.

Different pixel proportions do not prove that the physical sheet is smaller.
They can also result from auto-cropping, scanner borders, nonuniform resizing,
rotation, or a different paper format. Even an A4-shaped image does not prove
that its bounds are the physical paper edges. Actual A4 originals are therefore
also affected if the capture's scale, origin, or geometry is wrong.

The previous upload bug required A4 proportions or a detectable outer boundary
on all images. It is fixed: scanner exports can use image bounds, photographs
use boundary detection, and usable calibration markers retain the original
marker alignment. Do not reintroduce that rejection of ordinary scans.

## Current code and limits

- `src/printgpt_alignment/detection.py`: `detect_and_rectify_document` chooses
  marker, scan-frame, or page-boundary alignment. Non-A4 scans are uniformly
  fitted, with physical-size warnings. The reported confidence is heuristic;
  it is not a measured print-registration accuracy.
- `src/printgpt_alignment/document.py`: PDF rendering and raster loading.
  The current array-based handoff does not retain physical-page metadata for
  deciding whether the source can be registered to the original sheet.
- `src/printgpt_mvp/pipeline.py`: canonicalization, masks, generation, approval,
  and toner-only A4 export. Added canvas padding is excluded from the allowed
  region. `alignment.json` saves the chosen method, transform, and warnings.
- `src/printgpt_mvp/streamlit_app.py`: Automatic / Scanned page / Photo input
  modes. Input changes and failed runs clear previous previews and exports.
- The generator and critic are deterministic demonstration providers. The
  isolated diagonal line in the preview is a scripted sample addition, not a
  paper-size error or an intelligent document completion. Connecting a live
  semantic generator is a separate task from physical registration.
- The alignment notebook's synthetic and SmartDoc benchmarks overlay calibration
  markers on the source documents. Passing those cases does not establish
  accuracy for raw marker-free scans or a physical printer.

## Work to carry out

1. Trace physical coordinates from input through canonicalization, generation,
   validation, and PDF export. Define when source scale and paper-edge origin
   are known, inferred, or unknown, and retain that evidence in the run record.
2. Use trustworthy physical metadata where available: PDF page dimensions,
   rotation/crop information, and scanner DPI. Check consistency rather than
   blindly trusting metadata. For photographs, evaluate paper-edge detection
   and perspective correction. Provide a practical way to specify actual paper
   size or correct page corners when automatic evidence is insufficient.
3. Keep arbitrary scans usable for previews, but do not represent fitting an
   image to A4 as verified registration to the original sheet. Make the export
   behavior and any unresolved physical-size assumption clear to the user.
   Request a rescan only for a demonstrated capture problem, with its cause.
4. Establish printer calibration for the Canon imageRUNNER 2425i: horizontal and
   vertical offsets, rotation/skew, scale, feed variation, margins, and repeated
   runs. Use Actual Size / 100% with driver scaling disabled. Keep printer
   correction separate from validated source geometry and protection masks.
5. Verify end to end using ordinary A4 scans, scans with border padding and
   auto-cropping, rotated scans, non-A4 originals, PDFs, and perspective photos.
   Include bad captures that should reject. Measure displacement at independent
   locations across the page, not only the points used to fit the transform.
6. Add focused regression tests, update the UI/docs, and provide a reproducible
   physical test procedure. State what has been proven digitally and what still
   needs real print-and-rescan measurements. If printer access is unavailable,
   complete the software and measurement tooling and report that limitation.

## Acceptance criteria

- A confirmed full A4 scan has correct physical scale and origin, without
  unintended fitting or padding shifts.
- Missing geometry is exposed rather than hidden behind a high quality score.
- Preview and printable additions use a documented, consistent transform;
  existing ink protection and the explicit approval gate remain effective.
- Tests cover both acceptance and geometry, including meaningful displacement
  tolerances. Printer accuracy claims require repeated measured physical runs;
  a chosen tolerance must be justified against the protection clearance.
- Ordinary scan uploads continue to work without markers. Blurred or unusable
  captures still give actionable feedback.

## Validation already completed before this handoff

The current upload fix passed 156 tests, including the Streamlit upload flow;
one optional Gemini test was skipped because its SDK was absent. Five distinct
saved documents (four raster scans and one PDF), plus the notebook's marked
example, each produced two validated candidates. These checks establish
software processing, not physical print accuracy. The private local audit is
under the previous chat worktree's ignored `artifacts/upload_debug/report.json`;
the uploaded scans are not committed to the repository.

Restart Streamlit after changing imported modules, and verify which checkout
the running process imports. A screen rerun previously retained stale imported
pipeline functions.
