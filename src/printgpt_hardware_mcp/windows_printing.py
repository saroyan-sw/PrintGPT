"""Windows print-spooler adapter.

All pywin32 calls live in :class:`Win32PrinterBackend`. The rest of the package
depends only on the :class:`PrinterBackend` protocol so tests can use fakes and
never reach a physical device.

Printing method: the approved 300-DPI toner raster is drawn through GDI onto a
printer DC created with an explicit DEVMODE (A4, portrait, 1 copy, simplex,
monochrome, 100 %). The bitmap is placed so that raster pixel (0, 0) lands on
paper position (0, 0); the printer's printable-area offset is subtracted, and no
fit/shrink/centering is applied. PDFs are never sent raw to the UFR II queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import socket
from typing import Protocol

from PIL import Image

# DEVMODE field flags and values (wingdi.h).
DM_ORIENTATION = 0x00000001
DM_PAPERSIZE = 0x00000002
DM_SCALE = 0x00000010
DM_COPIES = 0x00000100
DM_COLOR = 0x00000800
DM_DUPLEX = 0x00001000
DMPAPER_A4 = 9
DMORIENT_PORTRAIT = 1
DMDUP_SIMPLEX = 1
DMCOLOR_MONOCHROME = 1

PAPER_NAMES = {1: "Letter", 5: "Legal", 8: "A3", 9: "A4", 11: "A5", 13: "B5"}
DUPLEX_NAMES = {1: "simplex", 2: "duplex long edge", 3: "duplex short edge"}
ORIENTATION_NAMES = {1: "portrait", 2: "landscape"}
COLOR_NAMES = {1: "monochrome", 2: "color"}

PRINTER_STATUS_FLAGS = {
    0x00000001: "paused",
    0x00000002: "error",
    0x00000004: "pending_deletion",
    0x00000008: "paper_jam",
    0x00000010: "paper_out",
    0x00000020: "manual_feed",
    0x00000040: "paper_problem",
    0x00000080: "offline",
    0x00000100: "io_active",
    0x00000200: "busy",
    0x00000400: "printing",
    0x00000800: "output_bin_full",
    0x00001000: "not_available",
    0x00002000: "waiting",
    0x00004000: "processing",
    0x00008000: "initializing",
    0x00010000: "warming_up",
    0x00020000: "toner_low",
    0x00040000: "no_toner",
    0x00080000: "page_punt",
    0x00100000: "user_intervention",
    0x00200000: "out_of_memory",
    0x00400000: "door_open",
    0x00800000: "server_unknown",
    0x01000000: "power_save",
}
BLOCKING_PRINTER_STATES = frozenset(
    {"paused", "error", "pending_deletion", "paper_jam", "paper_out", "paper_problem", "offline",
     "not_available", "no_toner", "user_intervention", "out_of_memory", "door_open", "server_unknown"}
)
PRINTER_ATTRIBUTE_WORK_OFFLINE = 0x00000400

JOB_STATUS_FLAGS = {
    0x0001: "paused",
    0x0002: "error",
    0x0004: "deleting",
    0x0008: "spooling",
    0x0010: "printing",
    0x0020: "offline",
    0x0040: "paper_out",
    0x0080: "printed",
    0x0100: "deleted",
    0x0200: "blocked_devq",
    0x0400: "user_intervention",
    0x0800: "restart",
    0x1000: "complete",
    0x2000: "retained",
}


def decode_flags(value: int, table: dict[int, str]) -> list[str]:
    return [name for bit, name in table.items() if value & bit]


@dataclass(frozen=True, slots=True)
class PrinterRecord:
    name: str
    driver: str
    port: str
    status_flags: int
    attributes: int
    job_count: int
    default_paper: int | None = None
    default_duplex: int | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: int
    document: str
    status_flags: int
    status_text: str | None = None
    pages_printed: int | None = None
    total_pages: int | None = None
    submitted_at: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    papers: list[tuple[int, str, float, float]]
    resolutions: list[tuple[int, int]]
    color: bool
    duplex: bool
    max_copies: int | None
    a4_device: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    job_id: int
    effective: dict[str, int | None]
    device_geometry: dict[str, int]
    notes: list[str] = field(default_factory=list)


class PrintBackendError(RuntimeError):
    """Raised by a backend for any spooler/driver failure."""


class PrinterBackend(Protocol):
    """The minimal Windows spooler surface the server needs."""

    def list_printers(self) -> list[PrinterRecord]: ...
    def default_printer(self) -> str | None: ...
    def port_host_address(self, port: str) -> str | None: ...
    def driver_version(self, driver: str) -> str | None: ...
    def capabilities(self, queue: str) -> CapabilityRecord: ...
    def list_jobs(self, queue: str) -> list[JobRecord]: ...
    def submit_raster(self, queue: str, image: Image.Image, document_name: str) -> SubmissionRecord: ...
    def cancel_job(self, queue: str, job_id: int) -> None: ...


def tcp_reachable(host: str, port: int = 9100, timeout: float = 1.5) -> bool:
    """A TCP connect to the RAW print port; sends no data."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def required_devmode_values() -> dict[str, int]:
    return {
        "PaperSize": DMPAPER_A4,
        "Orientation": DMORIENT_PORTRAIT,
        "Copies": 1,
        "Duplex": DMDUP_SIMPLEX,
        "Color": DMCOLOR_MONOCHROME,
        "Scale": 100,
    }


def a4_geometry_ok(geometry: dict[str, int], tolerance_mm: float = 1.0) -> tuple[bool, str]:
    """Check that a printer DC really describes a portrait A4 sheet."""
    dpi_x, dpi_y = geometry.get("LOGPIXELSX", 0), geometry.get("LOGPIXELSY", 0)
    if dpi_x <= 0 or dpi_y <= 0:
        return False, "driver did not report device resolution"
    width = geometry.get("PHYSICALWIDTH", 0) * 25.4 / dpi_x
    height = geometry.get("PHYSICALHEIGHT", 0) * 25.4 / dpi_y
    ok = abs(width - 210.0) <= tolerance_mm and abs(height - 297.0) <= tolerance_mm
    return ok, f"device page {width:.2f} x {height:.2f} mm at {dpi_x}x{dpi_y} DPI"


def placement_rect(geometry: dict[str, int]) -> tuple[int, int, int, int]:
    """Device rectangle covering the whole physical sheet (origin = printable area)."""
    off_x, off_y = geometry["PHYSICALOFFSETX"], geometry["PHYSICALOFFSETY"]
    return (-off_x, -off_y, geometry["PHYSICALWIDTH"] - off_x, geometry["PHYSICALHEIGHT"] - off_y)


class Win32PrinterBackend:
    """Real implementation over pywin32. Imported lazily so non-Windows tests work."""

    _CAPS = {
        "PHYSICALWIDTH": 110, "PHYSICALHEIGHT": 111, "PHYSICALOFFSETX": 112,
        "PHYSICALOFFSETY": 113, "HORZRES": 8, "VERTRES": 10, "LOGPIXELSX": 88, "LOGPIXELSY": 90,
    }

    def __init__(self) -> None:
        import win32print  # noqa: F401  (fail fast if pywin32 is missing)

    # ------------------------------------------------------------ queries
    def list_printers(self) -> list[PrinterRecord]:
        import win32print

        records = []
        flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
        for info in win32print.EnumPrinters(flags, None, 2):
            devmode = info.get("pDevMode")
            records.append(
                PrinterRecord(
                    name=info["pPrinterName"],
                    driver=info["pDriverName"],
                    port=info["pPortName"],
                    status_flags=int(info["Status"]),
                    attributes=int(info["Attributes"]),
                    job_count=int(info["cJobs"]),
                    default_paper=None if devmode is None else int(devmode.PaperSize),
                    default_duplex=None if devmode is None else int(devmode.Duplex),
                )
            )
        return records

    def default_printer(self) -> str | None:
        import win32print

        try:
            return win32print.GetDefaultPrinter()
        except Exception:
            return None

    def port_host_address(self, port: str) -> str | None:
        try:
            import win32com.client

            wmi = win32com.client.GetObject(r"winmgmts:\\.\root\cimv2")
            safe = port.replace("\\", "\\\\").replace("'", "\\'")
            for item in wmi.ExecQuery(f"SELECT HostAddress FROM Win32_TCPIPPrinterPort WHERE Name='{safe}'"):
                return str(item.HostAddress)
        except Exception:
            return None
        return None

    def driver_version(self, driver: str) -> str | None:
        import win32print

        try:
            for info in win32print.EnumPrinterDrivers(None, None, 6):
                if info["Name"] == driver:
                    packed = int(info["DriverVersion"])
                    parts = [(packed >> shift) & 0xFFFF for shift in (48, 32, 16, 0)]
                    return ".".join(str(part) for part in parts)
        except Exception:
            return None
        return None

    def capabilities(self, queue: str) -> CapabilityRecord:
        import win32con
        import win32print

        info = self._printer_info(queue)
        port = info["pPortName"]
        caps = win32print.DeviceCapabilities
        ids = caps(queue, port, win32con.DC_PAPERS)
        names = caps(queue, port, win32con.DC_PAPERNAMES)
        sizes = caps(queue, port, win32con.DC_PAPERSIZE)
        papers = []
        for paper_id, name, size in zip(ids, names, sizes):
            width, height = (size["x"], size["y"]) if isinstance(size, dict) else size
            papers.append((int(paper_id), str(name).strip("\x00"), width / 10.0, height / 10.0))
        resolutions = [(int(r["xdpi"]), int(r["ydpi"])) for r in caps(queue, port, win32con.DC_ENUMRESOLUTIONS)]
        copies = caps(queue, port, win32con.DC_COPIES)
        return CapabilityRecord(
            papers=papers,
            resolutions=resolutions,
            color=bool(caps(queue, port, win32con.DC_COLORDEVICE)),
            duplex=bool(caps(queue, port, win32con.DC_DUPLEX)),
            max_copies=int(copies) if copies and copies > 0 else None,
            a4_device=self._a4_geometry(queue),
        )

    def list_jobs(self, queue: str) -> list[JobRecord]:
        import win32print

        handle = win32print.OpenPrinter(queue)
        try:
            jobs = win32print.EnumJobs(handle, 0, 999, 1)
        finally:
            win32print.ClosePrinter(handle)
        records = []
        for job in jobs:
            submitted = job.get("Submitted")
            records.append(
                JobRecord(
                    job_id=int(job["JobId"]),
                    document=str(job.get("pDocument") or ""),
                    status_flags=int(job.get("Status") or 0),
                    status_text=job.get("pStatus"),
                    pages_printed=job.get("PagesPrinted"),
                    total_pages=job.get("TotalPages"),
                    submitted_at=None if submitted is None else str(submitted),
                )
            )
        return records

    # ------------------------------------------------------------ actions
    def submit_raster(self, queue: str, image: Image.Image, document_name: str) -> SubmissionRecord:
        import ctypes

        import win32gui
        import win32print
        from PIL import ImageWin

        devmode = self._job_devmode(queue)
        effective = {key: int(getattr(devmode, key)) for key in required_devmode_values()}
        wrong = {k: v for k, v in effective.items() if v != required_devmode_values()[k]}
        if wrong:
            raise PrintBackendError(f"driver rejected required settings: {wrong}")
        hdc = win32gui.CreateDC("WINSPOOL", queue, devmode)
        if not hdc:
            raise PrintBackendError("could not create a printer device context")
        started = False
        try:
            geometry = {name: ctypes.windll.gdi32.GetDeviceCaps(int(hdc), index) for name, index in self._CAPS.items()}
            ok, detail = a4_geometry_ok(geometry)
            if not ok:
                raise PrintBackendError(f"printer context is not A4 portrait ({detail}); nothing was spooled")
            job_id = int(win32print.StartDoc(int(hdc), (document_name, None, None, 0)))
            started = True
            win32print.StartPage(int(hdc))
            ctypes.windll.gdi32.SetStretchBltMode(int(hdc), 3)  # COLORONCOLOR: no dithering blend
            ImageWin.Dib(image.convert("L")).draw(int(hdc), placement_rect(geometry))
            win32print.EndPage(int(hdc))
            win32print.EndDoc(int(hdc))
            started = False
        except PrintBackendError:
            raise
        except Exception as exc:
            raise PrintBackendError(f"spooler error: {exc}") from exc
        finally:
            if started:
                try:
                    win32print.AbortDoc(int(hdc))
                except Exception:
                    pass
            win32gui.DeleteDC(hdc)
        return SubmissionRecord(job_id=job_id, effective=effective, device_geometry=geometry)

    def cancel_job(self, queue: str, job_id: int) -> None:
        import win32print

        handle = win32print.OpenPrinter(queue)
        try:
            win32print.SetJob(handle, int(job_id), 0, None, win32print.JOB_CONTROL_DELETE)
        except Exception as exc:
            raise PrintBackendError(f"could not cancel job {job_id}: {exc}") from exc
        finally:
            win32print.ClosePrinter(handle)

    # ------------------------------------------------------------ helpers
    def _printer_info(self, queue: str) -> dict:
        import win32print

        handle = win32print.OpenPrinter(queue)
        try:
            return win32print.GetPrinter(handle, 2)
        finally:
            win32print.ClosePrinter(handle)

    def _job_devmode(self, queue: str):
        """Start from the queue's full DEVMODE and let the driver merge our overrides."""
        import win32print

        handle = win32print.OpenPrinter(queue)
        try:
            devmode = win32print.GetPrinter(handle, 2)["pDevMode"]
            for key, value in required_devmode_values().items():
                setattr(devmode, key, value)
            devmode.Fields |= DM_PAPERSIZE | DM_ORIENTATION | DM_COPIES | DM_DUPLEX | DM_COLOR | DM_SCALE
            result = win32print.DocumentProperties(0, handle, queue, devmode, devmode, 2 | 8)
            if result < 0:
                raise PrintBackendError("driver refused the job DEVMODE")
            return devmode
        finally:
            win32print.ClosePrinter(handle)

    def _a4_geometry(self, queue: str) -> dict[str, int] | None:
        import ctypes

        import win32gui

        try:
            devmode = self._job_devmode(queue)
            hdc = win32gui.CreateDC("WINSPOOL", queue, devmode)
        except Exception:
            return None
        try:
            return {name: ctypes.windll.gdi32.GetDeviceCaps(int(hdc), index) for name, index in self._CAPS.items()}
        finally:
            win32gui.DeleteDC(hdc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
