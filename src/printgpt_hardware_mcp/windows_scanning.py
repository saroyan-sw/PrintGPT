"""Scanner status, optional WIA acquisition, and the manual scan-import inbox.

The Canon Color Network ScanGear 2 driver exposes a WIA 2.0 device, but the
device only answers when the printer panel is in *Remote Scanner* mode. Direct
WIA acquisition is therefore opt-in (``PRINTGPT_HW_ENABLE_WIA_SCAN=1``) and
runs in a subprocess with a hard timeout. The always-available path is manual:
the user scans with ScanGear or Scan & Send, then imports an explicitly named
file from the configured scans directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable, Protocol


@dataclass(frozen=True, slots=True)
class WiaRecord:
    found: bool
    name: str | None = None
    driver_version: str | None = None
    device_online: bool | None = None
    optical_dpi: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AcquireResult:
    ok: bool
    output_path: Path | None = None
    message: str = ""
    notes: list[str] = field(default_factory=list)


class ScannerBackend(Protocol):
    def wia_status(self) -> WiaRecord: ...
    def twain_sources(self) -> list[str]: ...
    def acquire(self, output_path: Path, *, color: bool, timeout_seconds: float) -> AcquireResult: ...


class Win32ScannerBackend:
    """Reads WIA properties (no scan) and launches opt-in acquisition subprocesses."""

    NAME_HINT = "scangear"

    def wia_status(self) -> WiaRecord:
        try:
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            manager = win32com.client.Dispatch("WIA.DeviceManager")
            for index in range(1, manager.DeviceInfos.Count + 1):
                info = manager.DeviceInfos.Item(index)
                props = {info.Properties.Item(j).Name: info.Properties.Item(j).Value
                         for j in range(1, info.Properties.Count + 1)}
                name = str(props.get("Name") or "")
                if info.Type != 1 or self.NAME_HINT not in name.lower():
                    continue
                online = optical = None
                try:
                    device = info.Connect()
                    dprops = {device.Properties.Item(j).Name: device.Properties.Item(j).Value
                              for j in range(1, device.Properties.Count + 1)}
                    raw_online = dprops.get("ScanGear IR Device Online")
                    online = None if raw_online is None else bool(int(raw_online))
                    raw_dpi = dprops.get("Horizontal Optical Resolution")
                    optical = None if raw_dpi is None else int(raw_dpi)
                except Exception as exc:  # property read failed; still report the device
                    return WiaRecord(True, name, str(props.get("Driver Version")), None, None, f"connect failed: {exc}")
                return WiaRecord(True, name, str(props.get("Driver Version")), online, optical)
            return WiaRecord(False)
        except Exception as exc:
            return WiaRecord(False, error=f"WIA unavailable: {exc}")

    def twain_sources(self) -> list[str]:
        roots = [Path(os.environ.get("WINDIR", r"C:\Windows")) / name for name in ("twain_32", "twain_64")]
        found = []
        for root in roots:
            if root.is_dir():
                found.extend(sorted(path.name for path in root.rglob("*.ds")))
        return found

    def acquire(self, output_path: Path, *, color: bool, timeout_seconds: float) -> AcquireResult:
        command = [sys.executable, "-m", "printgpt_hardware_mcp.wia_acquire", str(output_path),
                   "--mode", "color" if color else "grayscale"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds,
                                       stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return AcquireResult(False, message=(
                f"scan did not finish within {timeout_seconds:.0f} s; check that the printer panel shows "
                "Remote Scanner / online, then retry or use manual import"))
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()[-1:] or ["unknown error"]
            return AcquireResult(False, message=f"WIA scan failed: {detail[0][:300]}")
        return AcquireResult(True, output_path, "scan acquired through WIA")


# --------------------------------------------------------------------- inbox


@dataclass(frozen=True, slots=True)
class InboxFile:
    path: Path
    size: int
    modified: datetime


class ScanInbox:
    """Filesystem-only logic for listing and importing completed scans."""

    def __init__(self, scans_dir: Path, extensions: frozenset[str], *,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.scans_dir = scans_dir
        self.extensions = extensions
        self._sleep = sleep
        self._monotonic = monotonic

    def list_since(self, since: datetime | None, limit: int = 20) -> list[InboxFile]:
        if not self.scans_dir.is_dir():
            return []
        files = []
        for entry in self.scans_dir.iterdir():
            if not entry.is_file() or entry.is_symlink() or entry.suffix.lower() not in self.extensions:
                continue
            stat = entry.stat()
            modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            if since is not None and modified <= since:
                continue
            files.append(InboxFile(entry, stat.st_size, modified))
        files.sort(key=lambda item: item.modified, reverse=True)
        return files[:limit]

    def wait_until_stable(self, path: Path, *, quiet_seconds: float, timeout_seconds: float) -> bool:
        """True once size and mtime stop changing for ``quiet_seconds``."""
        deadline = self._monotonic() + timeout_seconds
        last = None
        stable_since = None
        while True:
            try:
                stat = path.stat()
            except OSError:
                return False
            current = (stat.st_size, stat.st_mtime_ns)
            now = self._monotonic()
            if current == last and current[0] > 0:
                stable_since = stable_since if stable_since is not None else now
                if now - stable_since >= quiet_seconds:
                    return True
            else:
                stable_since = None
            last = current
            if now >= deadline:
                return False
            self._sleep(min(0.5, max(quiet_seconds / 2, 0.05)))
