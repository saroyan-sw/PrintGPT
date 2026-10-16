# PrintGPT

PrintGPT aligns an A4 scan, protects the page's existing ink, generates
two continuation candidates, validates and criticizes sanitized previews, and
exports only the explicitly approved new ink. The dependable MVP path is
offline and deterministic; the optional Gemini adapter remains separately
verifiable.

## Integrated offline MVP

Install the project and Streamlit interface:

```powershell
python -m pip install -e ".[app]"
```

Run an input-to-approved-PDF demonstration:

```powershell
printgpt-mvp-demo artifacts/mvp_example
```

Run the approval interface:

```powershell
python -m streamlit run src/printgpt_mvp/streamlit_app.py
```

The interface has two explicit, non-fallback input modes:

- **Scanner A4** is for a flat, full-page A4 scan rendered at 300 DPI. It accepts
  portrait or landscape A4 within documented tolerances, provides Auto plus four
  manual rotation choices, normalizes deterministically to 2480 × 3508 without
  silently cropping, and requires confirmation of the corrected page preview.
  Auto applies the Canon imageRUNNER landscape profile (90° counter-clockwise).
- **Marked page (ArUco)** preserves the perspective-correction path for marked
  calibration sheets and photographs. All four expected markers are required;
  failed marker detection never switches to Scanner A4 mode.

The interface requires an explicit allowed completion rectangle, displays both model candidates and the actual sanitized
print previews, and creates no printable output until the user approves an
eligible candidate. The export is a toner-only PNG and a 210 × 297 mm PDF.
The offline interface visibly uses deterministic generator and critic doubles;
it does not claim to make live Gemini calls. A non-A4 image is rejected in
Scanner A4 mode rather than being assigned calibrated physical coordinates.
Missing or cropped content cannot be recovered, so verify that the entire
original page is present in the normalized preview.
Print the PDF at **100% / Actual Size** with Fit, Shrink, borderless expansion,
and automatic rotation disabled. See [`docs/mvp.md`](docs/mvp.md) for the
end-to-end contract and physical-printer limitations.

Scans can also arrive by email: choose **Scan source → Scanner email (Gmail)**
and press **Start waiting for scan**, then use the Canon's Scan and Send. The
app imports the newest scanner attachment with read-only Gmail access and
shows the same corrected A4 preview. Setup is in
[`docs/GMAIL_SCAN_INGESTION.md`](docs/GMAIL_SCAN_INGESTION.md) (install
`.[app,gmail]`); manual upload stays available.

The next physical-registration task is specified in
[`docs/PHYSICAL_ALIGNMENT_HANDOFF.md`](docs/PHYSICAL_ALIGNMENT_HANDOFF.md).

## Guided notebook

Open [`notebooks/03_mvp_agentic_system.ipynb`](notebooks/03_mvp_agentic_system.ipynb)
for the executed, visual explanation of the complete agentic MVP. It shows the
bounded LangGraph loop, protection masks, rejection and revision path, both
candidate images, sanitized approval previews, safety assertions, approval
gate, and exact-size export proof.

Open [`notebooks/01_a4_alignment_pipeline.ipynb`](notebooks/01_a4_alignment_pipeline.ipynb)
for the visual, reproducible walkthrough. It shows the calibration layout,
marker detection, rectification, independent ground-truth overlay, 30-case
synthetic benchmark, SmartDoc-derived benchmark, automated tests, and a cell
for trying your own PDF, scan, or photograph.

The notebook uses the registered **PrintGPT (.venv)** kernel. To recreate its
notebook dependencies in a fresh environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name printgpt --display-name=PrintGPT
```

## Commands

```powershell
.\.venv\Scripts\python.exe -m printgpt_alignment calibration
.\.venv\Scripts\python.exe -m printgpt_alignment rectify scan.pdf --page 1 --dpi 300 --output rectified.png --report detection.json
.\.venv\Scripts\python.exe -m printgpt_alignment rectify photo.jpg --output rectified.png
.\.venv\Scripts\python.exe -m printgpt_alignment generate
.\.venv\Scripts\python.exe -m printgpt_alignment benchmark
.\.venv\Scripts\python.exe -m printgpt_alignment smartdoc C:\path\to\sampleDataset.tar.gz
```

PDF input is rendered to pixels at the requested DPI before alignment. The
legacy alignment CLI still exposes its lower-level `--input-kind auto|scan|photo`
options; the MVP web interface instead uses the two explicit modes above.
The default is 300 DPI, matching the canonical 2480 × 3508 A4 coordinate
system. `--page` is 1-based and defaults to the first page. Rectified output is
a raster image because OpenCV's alignment calculations operate on pixels.

When running without installing the package, set `PYTHONPATH=src` first.
Run from this repository root. After changing imported pipeline modules, restart
the Streamlit server; a screen rerun can retain old imported functions.
Print the calibration PDF at actual size (100%) with all fit-to-page options disabled.

Component details: [masking](docs/masking.md),
[generation and criticism](docs/generation.md), and
[parallel-development contract](docs/PARALLEL_DEVELOPMENT.md).
