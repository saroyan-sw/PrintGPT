# PrintGPT A4 Alignment

This package establishes a physical A4 coordinate system, renders a printable
four-marker calibration page, detects and rectifies captures, generates
deterministic synthetic cases, and evaluates alignment in pixels and
millimetres.

## Guided notebook

Open [`notebooks/01_a4_alignment_pipeline.ipynb`](notebooks/01_a4_alignment_pipeline.ipynb)
for the visual, reproducible walkthrough. It shows the calibration layout,
marker detection, rectification, independent ground-truth overlay, 30-case
synthetic benchmark, SmartDoc-derived benchmark, automated tests, and a cell
for trying your own scan or photograph.

The notebook uses the registered **PrintGPT (.venv)** kernel. To recreate its
notebook dependencies in a fresh environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name printgpt --display-name=PrintGPT
```

## Commands

```powershell
.\.venv\Scripts\python.exe -m printgpt_alignment calibration
.\.venv\Scripts\python.exe -m printgpt_alignment rectify scan.jpg --output rectified.png --report detection.json
.\.venv\Scripts\python.exe -m printgpt_alignment generate
.\.venv\Scripts\python.exe -m printgpt_alignment benchmark
.\.venv\Scripts\python.exe -m printgpt_alignment smartdoc C:\path\to\sampleDataset.tar.gz
```

When running without installing the package, set `PYTHONPATH=src` first.
Print the calibration PDF at actual size (100%) with all fit-to-page options disabled.
