"""Hardware operations behind the MCP tools, with all dependencies injected."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .artifact_validation import file_sha256, resolve_allowed, validate_artifact
from .audit import AuditLog, JobLedger
from .config import HardwareConfig
from .models import (
    ArtifactValidation,
    CancelResult,
    HardwareError,
    ImportedScan,
    PaperSize,
    PreparedPrintJob,
    PrintableArea,
    PrinterCapabilities,
    PrinterStatus,
    PrinterSummary,
    PrintJobInfo,
    PrintSettings,
    ScanCandidate,
    ScannerStatus,
    SubmittedPrintJob,
)
from .windows_printing import (
    BLOCKING_PRINTER_STATES,
    COLOR_NAMES,
    DUPLEX_NAMES,
    JOB_STATUS_FLAGS,
    ORIENTATION_NAMES,
    PAPER_NAMES,
    PRINTER_ATTRIBUTE_WORK_OFFLINE,
    PRINTER_STATUS_FLAGS,
    PrintBackendError,
    PrinterBackend,
    a4_geometry_ok,
    decode_flags,
    tcp_reachable,
)
from .windows_scanning import ScanInbox, ScannerBackend

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
DOCUMENT_PREFIX = "PrintGPT"
BLOCKING_LEDGER_STATES = {"submitted", "left_queue"}

UNCONFIRMED_SETTINGS = [
    "Canon driver-private layout options (for example Enlarge Print Region, gutter, booklet or poster) "
    "stay at the queue defaults; they are not readable through the public DEVMODE",
    "the paper loaded in the tray the device picks is really A4",
    "physical scale and registration on paper (only a measured calibration print can confirm this)",
]


@dataclass
class _Preparation:
    preparation_id: str
    artifact_path: str
    artifact_sha256: str
    run_id: str
    candidate_id: str
    expires_at: datetime
    used: bool = False


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_timestamp(value: str | None, *, field: str) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise HardwareError("invalid_timestamp", f"{field} must be an ISO-8601 timestamp, e.g. 2026-10-17T14:30:00+04:00") from exc
    return parsed if parsed.tzinfo else parsed.astimezone()  # naive -> local time of this laptop


class HardwareService:
    def __init__(self, config: HardwareConfig, *, printer: PrinterBackend | None, scanner: ScannerBackend | None,
                 reachable: Callable[[str], bool] = tcp_reachable,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 inbox: ScanInbox | None = None) -> None:
        self.config = config
        self.printer = printer
        self.scanner = scanner
        self._reachable = reachable
        self._now = now
        self.audit = AuditLog(config.audit_log_path)
        self.ledger = JobLedger(config.job_ledger_path)
        self.inbox = inbox or ScanInbox(config.scans_dir, config.scan_extensions)
        self._preparations: dict[str, _Preparation] = {}
        self._lock = threading.Lock()
        self._margins_cache: tuple[float, float, float, float] | None = None

    # ================================================================ printers
    def _require_printer(self) -> PrinterBackend:
        if self.printer is None:
            raise HardwareError("printer_unavailable", "Windows printing (pywin32) is not available on this machine")
        return self.printer

    def list_printers(self) -> list[PrinterSummary]:
        backend = self._require_printer()
        default = backend.default_printer()
        return [PrinterSummary(name=r.name, driver=r.driver, port=r.port, is_default=r.name == default,
                               allowlisted=r.name == self.config.printer_queue) for r in backend.list_printers()]

    def _check_queue(self, queue: str | None) -> str:
        if queue is not None and queue != self.config.printer_queue:
            raise HardwareError("queue_not_allowlisted", f"only the queue {self.config.printer_queue!r} may be used")
        return self.config.printer_queue

    def printer_status(self, queue: str | None = None) -> PrinterStatus:
        queue = self._check_queue(queue)
        if self.printer is None:
            return PrinterStatus(queue=queue, found=False, allowlisted=True, notes=["pywin32 printing is unavailable"])
        try:
            record = next((r for r in self.printer.list_printers() if r.name == queue), None)
        except Exception as exc:
            return PrinterStatus(queue=queue, found=False, allowlisted=True, notes=[f"spooler query failed: {exc}"])
        if record is None:
            return PrinterStatus(queue=queue, found=False, allowlisted=True,
                                 notes=["the allowlisted queue is not installed; check the exact queue name"])
        host = self.printer.port_host_address(record.port)
        if host is None:
            try:
                host = str(ipaddress.ip_address(record.port))
            except ValueError:
                host = None
        port_ok = host == self.config.printer_ip
        reachable = self._reachable(self.config.printer_ip)
        states = decode_flags(record.status_flags, PRINTER_STATUS_FLAGS)
        if record.attributes & PRINTER_ATTRIBUTE_WORK_OFFLINE:
            states.append("work_offline")
        if reachable is False and "offline" not in states:
            states.append("network_unreachable")
        notes = ["Windows reports 'ready' for RAW TCP ports even when the device is off; the TCP probe is the stronger signal"]
        if not port_ok:
            notes.append(f"queue port resolves to {host!r}, not the allowlisted {self.config.printer_ip}")
        jobs = self._refresh_jobs(queue)
        online = reachable and not ({"offline", "work_offline", "not_available"} & set(states))
        ready = bool(online and port_ok and not (BLOCKING_PRINTER_STATES & set(states)))
        return PrinterStatus(
            queue=queue, found=True, allowlisted=True, driver=record.driver,
            driver_version=self.printer.driver_version(record.driver), port=record.port,
            port_host_address=host, port_matches_allowlist=port_ok, network_reachable=reachable,
            online=bool(online), ready=ready, states=states or ["idle"], queued_jobs=len(jobs),
            default_paper_size=PAPER_NAMES.get(record.default_paper, str(record.default_paper)),
            default_duplex=DUPLEX_NAMES.get(record.default_duplex, str(record.default_duplex)),
            notes=notes,
        )

    def printer_capabilities(self) -> PrinterCapabilities:
        backend = self._require_printer()
        queue = self.config.printer_queue
        record = backend.capabilities(queue)
        area = None
        if record.a4_device:
            g = record.a4_device
            ok, _ = a4_geometry_ok(g)
            if ok:
                dx, dy = g["LOGPIXELSX"], g["LOGPIXELSY"]
                to_mm = lambda value, dpi: round(value * 25.4 / dpi, 2)  # noqa: E731
                left, top = to_mm(g["PHYSICALOFFSETX"], dx), to_mm(g["PHYSICALOFFSETY"], dy)
                width, height = to_mm(g["HORZRES"], dx), to_mm(g["VERTRES"], dy)
                pw, ph = to_mm(g["PHYSICALWIDTH"], dx), to_mm(g["PHYSICALHEIGHT"], dy)
                area = PrintableArea(paper="A4", device_dpi=(dx, dy), physical_size_mm=(pw, ph),
                                     printable_offset_mm=(left, top), printable_size_mm=(width, height),
                                     margins_mm=(left, top, round(pw - left - width, 2), round(ph - top - height, 2)))
                self._margins_cache = area.margins_mm
        return PrinterCapabilities(
            queue=queue,
            paper_sizes=[PaperSize(id=i, name=n, width_mm=w, height_mm=h) for i, n, w, h in record.papers],
            resolutions_dpi=record.resolutions, color_supported=record.color, duplex_supported=record.duplex,
            max_copies=record.max_copies, a4_printable_area=area,
        )

    def _printer_margins(self) -> tuple[float, float, float, float] | None:
        if self._margins_cache is None and self.printer is not None:
            try:
                self.printer_capabilities()
            except Exception:
                return None
        return self._margins_cache

    def _refresh_jobs(self, queue: str) -> list[PrintJobInfo]:
        """List queue jobs and close ledger entries the spooler no longer lists."""
        records = self.printer.list_jobs(queue) if self.printer else []
        ours = {(j["windows_job_id"], j.get("document_name")): j for j in self.ledger.jobs() if j.get("queue") == queue}
        present = set()
        jobs = []
        for r in records:
            mine = (r.job_id, r.document) in ours
            if mine:
                present.add((r.job_id, r.document))
            jobs.append(PrintJobInfo(job_id=r.job_id, document=r.document if mine else "(not created by PrintGPT)",
                                     status=decode_flags(r.status_flags, JOB_STATUS_FLAGS) or ([r.status_text] if r.status_text else ["queued"]),
                                     pages_printed=r.pages_printed, total_pages=r.total_pages,
                                     submitted_at=r.submitted_at, created_by_printgpt=mine))
        for key, job in ours.items():
            if job.get("state") == "submitted" and key not in present:
                self.ledger.update(job["windows_job_id"], queue, state="left_queue", left_queue_at=_iso(self._now()))
                self.audit.record("completed", "left_queue", windows_job_id=job["windows_job_id"], printer_queue=queue,
                                  run_id=job.get("run_id"), artifact_sha256=job.get("artifact_sha256"),
                                  error="spooler no longer lists the job; physical output is not verified")
        return jobs

    def list_print_jobs(self) -> list[PrintJobInfo]:
        self._require_printer()
        return self._refresh_jobs(self.config.printer_queue)

    # ================================================================ scanner
    def scanner_status(self) -> ScannerStatus:
        notes = []
        wia = self.scanner.wia_status() if self.scanner else None
        twain = self.scanner.twain_sources() if self.scanner else []
        reachable = self._reachable(self.config.printer_ip)
        if wia is None:
            notes.append("scanner adapter unavailable on this machine")
        elif not wia.found:
            notes.append(wia.error or "no Canon ScanGear WIA device found")
        elif wia.device_online is False:
            notes.append("ScanGear reports the device offline: select Scan > Remote Scanner on the printer panel before a direct scan")
        direct = bool(self.config.enable_wia_scan and wia and wia.found)
        notes.append("direct WIA acquisition has not been verified on this device; manual import is the supported path"
                     if not direct else "direct WIA scan is enabled but experimental; manual import remains the fallback")
        return ScannerStatus(
            wia_device_found=bool(wia and wia.found), wia_device_name=wia.name if wia else None,
            wia_driver_version=wia.driver_version if wia else None, twain_sources=twain,
            device_reports_online=wia.device_online if wia else None,
            optical_resolution_dpi=wia.optical_dpi if wia else None, network_reachable=reachable,
            direct_scan_enabled=direct, direct_scan_verified=False,
            acquisition_mode="wia_direct" if direct else "manual_import",
            scans_dir=str(self.config.scans_dir), scans_dir_exists=self.config.scans_dir.is_dir(), notes=notes,
        )

    def find_recent_scans(self, since: str, limit: int = 20) -> list[ScanCandidate]:
        cutoff = parse_timestamp(since, field="since")
        if cutoff is None:
            raise HardwareError("since_required", "pass the time the scan was started so old scans are never offered")
        limit = max(1, min(int(limit), 100))
        return [ScanCandidate(filename=item.path.name, size_bytes=item.size, modified_at=_iso(item.modified),
                              extension=item.path.suffix.lower()) for item in self.inbox.list_since(cutoff, limit)]

    @staticmethod
    def _run_id(run_id: str) -> str:
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise HardwareError("invalid_run_id", "run_id must be 1-64 letters, digits, '-' or '_'")
        return run_id

    def _scan_metadata(self, path: Path) -> dict[str, Any]:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                box = reader.pages[0].mediabox
                return {"format": "pdf", "page_count": len(reader.pages),
                        "page_size_mm": (round(float(box.width) * 25.4 / 72, 2), round(float(box.height) * 25.4 / 72, 2))}
            from PIL import Image

            with Image.open(path) as image:
                dpi = image.info.get("dpi")
                fmt = (image.format or suffix[1:]).lower()
                frames = getattr(image, "n_frames", 1)
                return {"format": fmt, "page_count": frames, "pixel_size": image.size,
                        "dpi": None if not dpi else (round(float(dpi[0]), 2), round(float(dpi[1]), 2)),
                        "page_size_mm": None if not dpi else (round(image.size[0] * 25.4 / float(dpi[0]), 2),
                                                               round(image.size[1] * 25.4 / float(dpi[1]), 2))}
        except Exception as exc:
            raise HardwareError("unreadable_scan", f"the scan could not be read as {suffix[1:].upper()}") from exc

    def _destination(self, run_id: str, filename: str) -> Path:
        stamp = self._now().strftime("%Y%m%dT%H%M%SZ")
        target = self.config.imports_root / run_id / "input" / f"{stamp}_{filename}"
        if target.exists():
            raise HardwareError("already_imported", "an import with this name already exists; wait a second and retry")
        return target

    def import_scan(self, filename: str, run_id: str, not_before: str | None = None) -> ImportedScan:
        run_id = self._run_id(run_id)
        if not isinstance(filename, str) or filename != Path(filename).name or filename in {"", ".", ".."} or "/" in filename or "\\" in filename:
            raise HardwareError("invalid_filename", "pass a bare filename exactly as returned by find_recent_scans")
        if Path(filename).suffix.lower() not in self.config.scan_extensions:
            raise HardwareError("unsupported_format", "only PDF, PNG, JPEG or TIFF scans can be imported")
        source = resolve_allowed(self.config.scans_dir / filename, [self.config.scans_dir], what="scan")
        cutoff = parse_timestamp(not_before, field="not_before")
        modified = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc)
        if cutoff is not None and modified <= cutoff:
            raise HardwareError("scan_too_old", "that scan was saved before the supplied start time")
        if source.stat().st_size > self.config.max_scan_bytes:
            raise HardwareError("scan_too_large", "scan exceeds the configured size limit")
        if not self.inbox.wait_until_stable(source, quiet_seconds=self.config.scan_stability_seconds,
                                            timeout_seconds=self.config.scan_stability_timeout_seconds):
            raise HardwareError("scan_still_writing", "the scan file is still changing; wait for the scanner to finish and retry")
        metadata = self._scan_metadata(source)
        target = self._destination(run_id, source.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        source_hash = file_sha256(source)
        shutil.copy2(source, target)
        if file_sha256(target) != source_hash:
            target.unlink(missing_ok=True)
            raise HardwareError("copy_mismatch", "the scan changed while it was being imported; retry")
        self.audit.record("scan_imported", "ok", run_id=run_id, scan_filename=source.name,
                          imported_path=str(target), artifact_sha256=source_hash)
        return ImportedScan(run_id=run_id, source_filename=source.name, imported_path=str(target), sha256=source_hash,
                            size_bytes=target.stat().st_size, next_step=(
                                "Open this file in the PrintGPT app (scanner A4 / registration). "
                                "It is an input only and can never be printed."), **metadata)

    def acquire_scan(self, run_id: str, color_mode: str = "grayscale") -> ImportedScan:
        run_id = self._run_id(run_id)
        if color_mode not in {"color", "grayscale"}:
            raise HardwareError("invalid_color_mode", "color_mode must be 'color' or 'grayscale'")
        if not self.config.enable_wia_scan or self.scanner is None:
            raise HardwareError("direct_scan_disabled", "direct scanning is disabled; scan with ScanGear or Scan & Send, then use find_recent_scans and import_scan")
        status = self.scanner.wia_status()
        if not status.found:
            raise HardwareError("scanner_not_found", "the Canon ScanGear WIA device was not found")
        if status.device_online is False:
            raise HardwareError("scanner_offline", "select Scan > Remote Scanner on the printer panel, place one A4 page, then retry")
        target = self._destination(run_id, "wia_scan.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        result = self.scanner.acquire(target, color=color_mode == "color", timeout_seconds=self.config.scan_timeout_seconds)
        if not result.ok or not target.is_file():
            self.audit.record("scan_acquired", "failed", run_id=run_id, error=result.message)
            raise HardwareError("scan_failed", result.message)
        metadata = self._scan_metadata(target)
        digest = file_sha256(target)
        self.audit.record("scan_acquired", "ok", run_id=run_id, imported_path=str(target), artifact_sha256=digest)
        return ImportedScan(run_id=run_id, source_filename=target.name, imported_path=str(target), sha256=digest,
                            size_bytes=target.stat().st_size, next_step="Open this file in the PrintGPT app as scanner input.",
                            **metadata)

    # ================================================================ printing
    def validate_print_artifact(self, artifact_path: str) -> ArtifactValidation:
        return validate_artifact(artifact_path, self.config, now=self._now(), printer_margins_mm=self._printer_margins()).report

    def _duplicate(self, sha: str) -> bool:
        return any(j.get("artifact_sha256") == sha and j.get("state") in BLOCKING_LEDGER_STATES for j in self.ledger.jobs())

    def prepare_print_job(self, artifact_path: str, copies: int = 1, queue: str | None = None) -> PreparedPrintJob:
        queue = self._check_queue(queue)
        if copies != 1:
            raise HardwareError("copies_rejected", "PrintGPT prints exactly one copy")
        if isinstance(artifact_path, (list, tuple)):
            raise HardwareError("batch_rejected", "batch printing is not supported; prepare one artifact")
        validated = validate_artifact(artifact_path, self.config, now=self._now(), printer_margins_mm=self._printer_margins())
        report = validated.report
        if not report.valid:
            self.audit.record("prepared", "rejected", artifact_path=report.artifact_path, artifact_sha256=report.sha256,
                              printer_queue=queue, error="; ".join(report.errors))
            raise HardwareError("artifact_rejected", "; ".join(report.errors))
        if self._duplicate(report.sha256):
            raise HardwareError("duplicate_submission", "this exact artifact was already submitted; cancel it or approve a new export")
        status = self.printer_status(queue)
        warnings = list(report.warnings)
        if not status.ready:
            warnings.append(f"printer is not ready ({', '.join(status.states)}); submission will be refused until it is")
        if not self.config.allow_physical_print:
            warnings.append("physical printing is disabled in this server (PRINTGPT_HW_ALLOW_PRINT is not 1); submit will only dry-run")
        now = self._now()
        preparation = _Preparation(uuid4().hex, report.artifact_path, report.sha256, report.run_id, report.candidate_id,
                                   now + timedelta(seconds=self.config.preparation_ttl_seconds))
        with self._lock:
            self._preparations[preparation.preparation_id] = preparation
        settings = PrintSettings()
        summary = (
            f"Would print ONE page, ONE copy, of {Path(report.artifact_path).name} "
            f"(run {report.run_id}, candidate {report.candidate_id}, SHA-256 {report.sha256[:16]}...) "
            f"to '{queue}' at {self.config.printer_ip}: A4 210 x 297 mm portrait, simplex, monochrome, 100% scale, "
            f"no fit-to-page, toner on {report.ink_coverage_percent:.3f}% of the page. "
            "Load the original sheet in the tray in the orientation recorded during calibration. "
            "Nothing has been printed. Printing requires an explicit user request and submit_print_job."
        )
        self.audit.record("prepared", "ok", preparation_id=preparation.preparation_id, run_id=report.run_id,
                          candidate_id=report.candidate_id, artifact_sha256=report.sha256, artifact_path=report.artifact_path,
                          printer_queue=queue, requested_settings=settings.model_dump())
        return PreparedPrintJob(
            preparation_id=preparation.preparation_id, expires_at=_iso(preparation.expires_at), queue=queue,
            printer_ip=self.config.printer_ip, artifact_path=report.artifact_path, artifact_sha256=report.sha256,
            run_id=report.run_id, candidate_id=report.candidate_id, page_count=1, page_size_mm=(210.0, 297.0), copies=1,
            settings=settings, physical_print_enabled=self.config.allow_physical_print, printer_ready=status.ready,
            warnings=warnings, summary=summary,
        )

    def submit_print_job(self, preparation_id: str, artifact_sha256: str, confirm_physical_print: bool,
                         dry_run: bool = True) -> SubmittedPrintJob:
        queue = self.config.printer_queue
        with self._lock:
            preparation = self._preparations.get(preparation_id)
            if preparation is None:
                raise HardwareError("unknown_preparation", "call prepare_print_job first and pass its preparation_id")
            if preparation.used:
                raise HardwareError("preparation_used", "this preparation was already submitted; prepare again")
            if self._now() > preparation.expires_at:
                raise HardwareError("preparation_expired", "the preparation expired; call prepare_print_job again")
            if artifact_sha256 != preparation.artifact_sha256:
                raise HardwareError("hash_mismatch", "artifact_sha256 does not match the prepared artifact")
            if confirm_physical_print is not True:
                raise HardwareError("confirmation_required", "set confirm_physical_print=true only after the user explicitly asked to print")
            if not dry_run:
                preparation.used = True
        validated = validate_artifact(preparation.artifact_path, self.config, now=self._now(), printer_margins_mm=self._printer_margins())
        report = validated.report
        settings = PrintSettings()
        document_name = f"{DOCUMENT_PREFIX} {preparation.run_id} {preparation.artifact_sha256[:12]}"
        base_audit = dict(preparation_id=preparation_id, run_id=preparation.run_id, candidate_id=preparation.candidate_id,
                          artifact_sha256=preparation.artifact_sha256, printer_queue=queue,
                          requested_settings=settings.model_dump(), document_name=document_name, dry_run=dry_run)
        if not report.valid or report.sha256 != preparation.artifact_sha256 or validated.raster is None:
            reason = "artifact changed since preparation (SHA-256 mismatch)" if report.sha256 != preparation.artifact_sha256 else "; ".join(report.errors)
            self.audit.record("submitted", "rejected", error=reason, **base_audit)
            raise HardwareError("artifact_rejected", reason)
        if self._duplicate(preparation.artifact_sha256):
            raise HardwareError("duplicate_submission", "this exact artifact was already submitted")
        status = self.printer_status(queue)
        if dry_run:
            self.audit.record("submitted", "dry_run", **base_audit)
            return SubmittedPrintJob(
                dry_run=True, windows_job_id=None, queue=queue, document_name=document_name,
                artifact_sha256=preparation.artifact_sha256, submitted_at=_iso(self._now()), requested_settings=settings,
                effective_settings={}, unconfirmed_settings=UNCONFIRMED_SETTINGS, queue_status=status.states,
                message="Dry run only: all checks passed and nothing was sent to the printer.",
            )
        if not self.config.allow_physical_print:
            self.audit.record("submitted", "rejected", error="physical printing disabled", **base_audit)
            raise HardwareError("printing_disabled", "physical printing is disabled (set PRINTGPT_HW_ALLOW_PRINT=1 and restart the server)")
        if not status.ready:
            self.audit.record("submitted", "rejected", error=f"printer not ready: {status.states}", **base_audit)
            raise HardwareError("printer_not_ready", f"printer is not ready: {', '.join(status.states)}")
        backend = self._require_printer()
        submitted_at = self._now()
        try:
            submission = backend.submit_raster(queue, validated.raster, document_name)
        except PrintBackendError as exc:
            self.audit.record("failed", "error", error=str(exc), **base_audit)
            raise HardwareError("print_failed", str(exc)) from exc
        effective = {
            "paper_size": PAPER_NAMES.get(submission.effective.get("PaperSize"), str(submission.effective.get("PaperSize"))),
            "orientation": ORIENTATION_NAMES.get(submission.effective.get("Orientation"), None),
            "copies": submission.effective.get("Copies"),
            "duplex": DUPLEX_NAMES.get(submission.effective.get("Duplex"), None),
            "color_mode": COLOR_NAMES.get(submission.effective.get("Color"), None),
            "scale_percent": submission.effective.get("Scale"),
            "device_dpi": f"{submission.device_geometry.get('LOGPIXELSX')}x{submission.device_geometry.get('LOGPIXELSY')}",
            "placement": "raster drawn at exact physical size from the paper's top-left corner (no fit, no centering)",
        }
        self.ledger.add({"windows_job_id": submission.job_id, "queue": queue, "document_name": document_name,
                         "artifact_sha256": preparation.artifact_sha256, "run_id": preparation.run_id,
                         "submitted_at": _iso(submitted_at), "state": "submitted"})
        self.audit.record("submitted", "ok", windows_job_id=submission.job_id, effective_settings=effective,
                          unconfirmed_settings=UNCONFIRMED_SETTINGS, **base_audit)
        queue_status = next((j.status for j in self._refresh_jobs(queue) if j.job_id == submission.job_id), ["left_queue"])
        return SubmittedPrintJob(
            dry_run=False, windows_job_id=submission.job_id, queue=queue, document_name=document_name,
            artifact_sha256=preparation.artifact_sha256, submitted_at=_iso(submitted_at), requested_settings=settings,
            effective_settings=effective, unconfirmed_settings=UNCONFIRMED_SETTINGS, queue_status=queue_status,
            message="Submitted one page to the Windows queue. Spooling is not proof of correct rendering; measure the output.",
        )

    def cancel_print_job(self, windows_job_id: int) -> CancelResult:
        queue = self.config.printer_queue
        backend = self._require_printer()
        entry = self.ledger.find(int(windows_job_id), queue)
        if entry is None:
            raise HardwareError("not_printgpt_job", "only jobs submitted by this PrintGPT server can be cancelled")
        live = next((j for j in backend.list_jobs(queue) if j.job_id == int(windows_job_id)), None)
        if live is None:
            return CancelResult(windows_job_id=int(windows_job_id), cancelled=False, message="job is no longer in the queue")
        if live.document != entry.get("document_name"):
            raise HardwareError("not_printgpt_job", "the queue job with this ID was not created by PrintGPT (ID reused)")
        try:
            backend.cancel_job(queue, int(windows_job_id))
        except PrintBackendError as exc:
            self.audit.record("cancelled", "error", windows_job_id=int(windows_job_id), printer_queue=queue, error=str(exc))
            raise HardwareError("cancel_failed", str(exc)) from exc
        self.ledger.update(int(windows_job_id), queue, state="cancelled", cancelled_at=_iso(self._now()))
        self.audit.record("cancelled", "ok", windows_job_id=int(windows_job_id), printer_queue=queue,
                          run_id=entry.get("run_id"), artifact_sha256=entry.get("artifact_sha256"))
        return CancelResult(windows_job_id=int(windows_job_id), cancelled=True,
                            message="cancel requested; pages already sent to the device may still print")

    # ================================================================ overview
    def hardware_capabilities(self) -> dict[str, Any]:
        result: dict[str, Any] = {"server": self.config.public_summary()}
        try:
            result["printer_status"] = self.printer_status().model_dump()
        except Exception as exc:
            result["printer_status"] = {"error": str(exc)}
        try:
            result["printer_capabilities"] = self.printer_capabilities().model_dump()
        except Exception as exc:
            result["printer_capabilities"] = {"error": str(exc)}
        try:
            result["scanner_status"] = self.scanner_status().model_dump()
        except Exception as exc:
            result["scanner_status"] = {"error": str(exc)}
        result["workflow"] = [
            "scan: user scans with ScanGear/Scan & Send -> find_recent_scans(since) -> import_scan(filename, run_id)",
            "print: approve in PrintGPT -> validate_print_artifact -> prepare_print_job -> user explicitly asks -> submit_print_job",
        ]
        return result


def default_service(config: HardwareConfig | None = None) -> HardwareService:
    """Build the real service; hardware adapters degrade to None off Windows."""
    config = config or HardwareConfig.from_env()
    printer = scanner = None
    try:
        from .windows_printing import Win32PrinterBackend

        printer = Win32PrinterBackend()
    except Exception:
        printer = None
    try:
        from .windows_scanning import Win32ScannerBackend

        scanner = Win32ScannerBackend()
    except Exception:
        scanner = None
    return HardwareService(config, printer=printer, scanner=scanner)
