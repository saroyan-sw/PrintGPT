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

The interface accepts a supported image or PDF with or without calibration
markers, requires an explicit allowed completion rectangle, displays both model candidates and the actual sanitized
print previews, and creates no printable output until the user approves an
eligible candidate. The export is a toner-only PNG and a 210 × 297 mm PDF.
Markers are used when available. Otherwise, Automatic mode recognizes scanner
exports where light paper reaches the image borders, including ordinary scan
proportions and narrow scanner bars. These scans need no external page outline.
Photographs use a visible paper boundary. Choose **Scanned page** to explicitly
use the full image, or **Photo of a page** to require boundary detection.
Blurry, too faint, or undersized scans still request a rescan. An image whose
proportions differ from A4 is fitted without stretching; the app explains that
its physical size is unknown and alignment on the original sheet is not assured.
Missing content cannot be recovered or reliably identified from an already
cropped scanner export. Verify that the full original page is present.
Print the PDF at **100% / Actual Size** with Fit, Shrink, borderless expansion,
and automatic rotation disabled. See [`docs/mvp.md`](docs/mvp.md) for the
end-to-end contract and physical-printer limitations.

Fitting alone is now **preview-only**: exporting onto the original sheet requires
confirmed physical geometry (or the known ArUco layout). PDF boxes/rotation and
scanner DPI are retained; complete-page, corner and measured-crop corrections
are available. Printer accuracy remains unmeasured until physically tested.
See [`docs/PHYSICAL_ALIGNMENT.md`](docs/PHYSICAL_ALIGNMENT.md) for the coordinate
contract, Canon 2425i calibration procedure and independent-point checks.

## Canon printer and scanner (local MCP server)

`printgpt_hardware_mcp` is a local stdio MCP server that lets Codex check the Canon
imageRUNNER 2425i queue and ScanGear scanner, import a finished scan, and prepare and
submit **one** approved toner-only A4 page with explicit confirmation. See
[`docs/HARDWARE_MCP.md`](docs/HARDWARE_MCP.md) for setup, Codex registration and the
safe printing workflow, and [`docs/HARDWARE_AUDIT.md`](docs/HARDWARE_AUDIT.md) for the
audited hardware facts.

## Guided notebook

Open [`notebooks/04_physical_alignment.ipynb`](notebooks/04_physical_alignment.ipynb)
for the new marker-free geometry benchmark, visual overlays, source-coordinate
inspection, manual corrections and printable measurement target. Existing
notebooks have been preserved.

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

PDF input is rendered to pixels at the requested DPI before alignment. ArUco
markers are preferred, followed by scan-frame or photograph-boundary alignment.
The CLI has the same `--input-kind auto|scan|photo` options as the interface.
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
