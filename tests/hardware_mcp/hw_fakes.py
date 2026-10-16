"""Fakes and fixtures for the hardware MCP tests. Nothing here touches a device."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from printgpt_hardware_mcp.config import HardwareConfig
from printgpt_hardware_mcp.service import HardwareService
from printgpt_hardware_mcp.windows_printing import CapabilityRecord, JobRecord, PrinterRecord, SubmissionRecord
from printgpt_hardware_mcp.windows_scanning import AcquireResult, ScanInbox, WiaRecord
from printgpt_mvp.print_approval import write_print_approval

QUEUE = "Canon Generic Plus UFR II"
IP = "10.1.3.175"
NOW = datetime(2026, 10, 17, 10, 0, tzinfo=timezone.utc)
A4_GEOMETRY = {"PHYSICALWIDTH": 4960, "PHYSICALHEIGHT": 7014, "PHYSICALOFFSETX": 120, "PHYSICALOFFSETY": 120,
               "HORZRES": 4720, "VERTRES": 6774, "LOGPIXELSX": 600, "LOGPIXELSY": 600}


@pytest.fixture(autouse=True)
def _never_touch_real_hardware(monkeypatch):
    """Any accidental use of the real Win32 adapters fails the test immediately."""
    from printgpt_hardware_mcp import windows_printing, windows_scanning

    def refuse(*_args, **_kwargs):
        raise AssertionError("tests must never use the real Windows printer/scanner")

    monkeypatch.setattr(windows_printing.Win32PrinterBackend, "submit_raster", refuse)
    monkeypatch.setattr(windows_printing.Win32PrinterBackend, "cancel_job", refuse)
    monkeypatch.setattr(windows_scanning.Win32ScannerBackend, "acquire", refuse)


@dataclass
class FakePrinter:
    printers: list[PrinterRecord] = field(default_factory=lambda: [
        PrinterRecord(QUEUE, "Canon Generic Plus UFR II", IP, 0, 0x0A40, 0, 1, 2),
        PrinterRecord("Microsoft Print to PDF", "Microsoft Print To PDF", "PORTPROMPT:", 0, 0, 0, 9, 1),
    ])
    jobs: list[JobRecord] = field(default_factory=list)
    host: str | None = IP
    submitted: list[tuple[str, tuple[int, int], str]] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    next_job_id: int = 41
    geometry: dict[str, int] = field(default_factory=lambda: dict(A4_GEOMETRY))

    def list_printers(self):
        return list(self.printers)

    def default_printer(self):
        return QUEUE

    def port_host_address(self, port):
        return self.host if port == IP else None

    def driver_version(self, driver):
        return "3.50.0.0"

    def capabilities(self, queue):
        return CapabilityRecord(papers=[(1, "Letter", 215.9, 279.4), (9, "A4", 210.0, 297.0)],
                                resolutions=[(1200, 1200), (600, 600)], color=False, duplex=True,
                                max_copies=9999, a4_device=self.geometry)

    def list_jobs(self, queue):
        return list(self.jobs)

    def submit_raster(self, queue, image, document_name):
        self.next_job_id += 1
        self.submitted.append((queue, image.size, document_name))
        self.jobs.append(JobRecord(self.next_job_id, document_name, 0x0008))
        return SubmissionRecord(self.next_job_id, {"PaperSize": 9, "Orientation": 1, "Copies": 1, "Duplex": 1,
                                                   "Color": 1, "Scale": 100}, dict(self.geometry))

    def cancel_job(self, queue, job_id):
        self.cancelled.append(job_id)
        self.jobs = [j for j in self.jobs if j.job_id != job_id]


@dataclass
class FakeScanner:
    record: WiaRecord = field(default_factory=lambda: WiaRecord(True, "Color Network ScanGear 2", "2.3.7.0", False, 300))
    acquired: list[Path] = field(default_factory=list)

    def wia_status(self):
        return self.record

    def twain_sources(self):
        return ["SGIR.DS", "wiatwain.ds"]

    def acquire(self, output_path, *, color, timeout_seconds):
        Image.new("L", (2480, 3508), 255).save(output_path, dpi=(300, 300))
        self.acquired.append(output_path)
        return AcquireResult(True, output_path, "ok")


@dataclass
class Clock:
    now: datetime = NOW

    def __call__(self):
        return self.now

    def advance(self, **kwargs):
        self.now += timedelta(**kwargs)


@pytest.fixture
def roots(tmp_path):
    artifacts = tmp_path / "artifacts"
    scans = tmp_path / "Scans"
    artifacts.mkdir()
    scans.mkdir()
    return artifacts, scans


@pytest.fixture
def config(roots):
    artifacts, scans = roots
    return HardwareConfig(artifacts_root=artifacts, scans_dir=scans, allow_physical_print=True,
                          scan_stability_seconds=0.0, scan_stability_timeout_seconds=1.0)


@pytest.fixture
def printer():
    return FakePrinter()


@pytest.fixture
def scanner():
    return FakeScanner()


@pytest.fixture
def clock():
    return Clock()


def make_service(config, printer, scanner, clock, reachable=True):
    inbox = ScanInbox(config.scans_dir, config.scan_extensions, sleep=lambda _s: None)
    return HardwareService(config, printer=printer, scanner=scanner, reachable=lambda _ip: reachable,
                           now=clock, inbox=inbox)


@pytest.fixture
def service(config, printer, scanner, clock):
    return make_service(config, printer, scanner, clock)


def toner_raster(size=(2480, 3508), *, gray=False) -> np.ndarray:
    pixels = np.full((size[1], size[0]), 255, np.uint8)
    pixels[1500:1520, 600:1800] = 0
    if gray:
        pixels[:] = np.random.default_rng(0).integers(150, 256, pixels.shape, dtype=np.uint8)
    return pixels


def write_export(run_dir: Path, pixels: np.ndarray | None = None, *, page_mm=(210.0, 297.0),
                 simulated=False, accepted=True, approved_at=NOW - timedelta(minutes=5),
                 manifest=True) -> dict[str, Path]:
    """Create an approved export exactly as printgpt_mvp writes it."""
    pixels = toner_raster() if pixels is None else pixels
    out = run_dir / "print"
    out.mkdir(parents=True, exist_ok=True)
    png = out / "approved_toner_only.png"
    Image.fromarray(pixels, mode="L").save(png, dpi=(300, 300))
    pdf = out / "approved_a4_actual_size.pdf"
    doc = canvas.Canvas(str(pdf), pagesize=(page_mm[0] * mm, page_mm[1] * mm))
    doc.drawImage(ImageReader(Image.fromarray(pixels, mode="L")), 0, 0, width=page_mm[0] * mm,
                  height=page_mm[1] * mm, preserveAspectRatio=False)
    doc.showPage()
    doc.save()
    paths = {"png": png, "pdf": pdf}
    if manifest:
        paths["manifest"] = write_print_approval(
            run_dir=run_dir, run_id="run-7", candidate_id="candidate-1", attempt_id="candidate-1/attempt-1",
            toner_png=png, a4_pdf=pdf, validation={"accepted": accepted, "simulated": simulated, "reasons": []},
            input_mode={"input_kind": "scan", "registration_status": "registered"},
            printer_correction_applied=False, approved_at=approved_at)
    return paths


def edit_manifest(path: Path, **changes) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        target = data
        parts = key.split("__")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def export(config):
    return write_export(config.artifacts_root / "streamlit" / "abc123" / "run")
