"""Typed, JSON-serialisable results returned by the hardware MCP tools."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HardwareError(Exception):
    """A safe, user-facing failure. Messages never contain file contents."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


# ---------------------------------------------------------------- printers


class PrinterSummary(_Model):
    name: str
    driver: str
    port: str
    is_default: bool = False
    allowlisted: bool = False


class PrintJobInfo(_Model):
    job_id: int
    document: str
    status: list[str]
    pages_printed: int | None = None
    total_pages: int | None = None
    submitted_at: str | None = None
    created_by_printgpt: bool = False


class PrinterStatus(_Model):
    queue: str
    found: bool
    allowlisted: bool
    driver: str | None = None
    driver_version: str | None = None
    port: str | None = None
    port_host_address: str | None = None
    port_matches_allowlist: bool = False
    network_reachable: bool | None = None
    online: bool = False
    ready: bool = False
    states: list[str] = Field(default_factory=list)
    queued_jobs: int = 0
    default_paper_size: str | None = None
    default_duplex: str | None = None
    notes: list[str] = Field(default_factory=list)


class PaperSize(_Model):
    id: int
    name: str
    width_mm: float
    height_mm: float


class PrintableArea(_Model):
    paper: str
    device_dpi: tuple[int, int]
    physical_size_mm: tuple[float, float]
    printable_offset_mm: tuple[float, float]
    printable_size_mm: tuple[float, float]
    margins_mm: tuple[float, float, float, float] = Field(description="left, top, right, bottom")


class PrinterCapabilities(_Model):
    queue: str
    paper_sizes: list[PaperSize]
    resolutions_dpi: list[tuple[int, int]]
    color_supported: bool
    duplex_supported: bool
    max_copies: int | None
    a4_printable_area: PrintableArea | None = None


# ---------------------------------------------------------------- scanner


class ScannerStatus(_Model):
    wia_device_found: bool
    wia_device_name: str | None = None
    wia_driver_version: str | None = None
    twain_sources: list[str] = Field(default_factory=list)
    device_reports_online: bool | None = Field(
        default=None,
        description="ScanGear 'Device Online' WIA property; false usually means Remote Scanner is not active on the printer panel.",
    )
    optical_resolution_dpi: int | None = None
    network_reachable: bool | None = None
    direct_scan_enabled: bool = False
    direct_scan_verified: bool = False
    acquisition_mode: Literal["wia_direct", "manual_import"] = "manual_import"
    scans_dir: str
    scans_dir_exists: bool
    notes: list[str] = Field(default_factory=list)


class ScanCandidate(_Model):
    filename: str
    size_bytes: int
    modified_at: str
    extension: str
    stable: bool | None = None


class ImportedScan(_Model):
    run_id: str
    source_filename: str
    imported_path: str
    sha256: str
    size_bytes: int
    format: str
    page_count: int | None = None
    pixel_size: tuple[int, int] | None = None
    page_size_mm: tuple[float, float] | None = None
    dpi: tuple[float, float] | None = None
    next_step: str


# ---------------------------------------------------------------- printing


class PrintSettings(_Model):
    paper_size: Literal["A4"] = "A4"
    orientation: Literal["portrait"] = "portrait"
    copies: int = 1
    duplex: Literal["simplex"] = "simplex"
    color_mode: Literal["monochrome"] = "monochrome"
    scale_percent: int = 100
    fit_to_page: bool = False


class ArtifactCheck(_Model):
    name: str
    passed: bool
    detail: str


class ArtifactValidation(_Model):
    valid: bool
    artifact_path: str | None = None
    manifest_path: str | None = None
    artifact_kind: Literal["png", "pdf"] | None = None
    sha256: str | None = None
    run_id: str | None = None
    candidate_id: str | None = None
    page_size_mm: tuple[float, float] | None = None
    pixel_size: tuple[int, int] | None = None
    ink_coverage_percent: float | None = None
    checks: list[ArtifactCheck] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class PreparedPrintJob(_Model):
    preparation_id: str
    expires_at: str
    queue: str
    printer_ip: str
    artifact_path: str
    artifact_sha256: str
    run_id: str
    candidate_id: str
    page_count: int
    page_size_mm: tuple[float, float]
    copies: int
    settings: PrintSettings
    physical_print_enabled: bool
    printer_ready: bool
    warnings: list[str]
    summary: str


class SubmittedPrintJob(_Model):
    dry_run: bool
    windows_job_id: int | None
    queue: str
    document_name: str
    artifact_sha256: str
    submitted_at: str
    requested_settings: PrintSettings
    effective_settings: dict[str, str | int | None]
    unconfirmed_settings: list[str]
    queue_status: list[str]
    message: str


class CancelResult(_Model):
    windows_job_id: int
    cancelled: bool
    message: str
