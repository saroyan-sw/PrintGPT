"""Scanner detection and the manual scan-import workflow (filesystem only)."""

from datetime import timedelta
import os

from PIL import Image
import pytest

from printgpt_hardware_mcp.config import HardwareConfig
from printgpt_hardware_mcp.models import HardwareError
from printgpt_hardware_mcp.windows_scanning import ScanInbox, WiaRecord

from hw_fakes import NOW, make_service


def _scan(scans, name, *, when=NOW, content=None):
    path = scans / name
    if content is None:
        Image.new("L", (2480, 3508), 250).save(path, dpi=(300, 300))
    else:
        path.write_bytes(content)
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def test_scanner_detection_reports_offline_remote_scanner(service):
    status = service.scanner_status()
    assert status.wia_device_found and status.device_reports_online is False
    assert status.optical_resolution_dpi == 300 and "SGIR.DS" in status.twain_sources
    assert status.acquisition_mode == "manual_import" and not status.direct_scan_verified
    assert any("Remote Scanner" in note for note in status.notes)


def test_missing_wia_device(service, scanner):
    scanner.record = WiaRecord(False, error="WIA unavailable")
    status = service.scanner_status()
    assert not status.wia_device_found and "WIA unavailable" in status.notes


def test_recent_scans_require_since_and_exclude_older_files(service, config):
    _scan(config.scans_dir, "old.pdf", when=NOW - timedelta(days=1), content=b"%PDF-1.4")
    _scan(config.scans_dir, "new.png", when=NOW + timedelta(minutes=1))
    _scan(config.scans_dir, "notes.docx", when=NOW + timedelta(minutes=1), content=b"x")
    with pytest.raises(HardwareError, match="since"):
        service.find_recent_scans("")
    names = [c.filename for c in service.find_recent_scans(NOW.isoformat())]
    assert names == ["new.png"]


def test_manual_import_copies_into_run_input_with_metadata(service, config):
    _scan(config.scans_dir, "0776_001.png", when=NOW + timedelta(minutes=1))
    imported = service.import_scan("0776_001.png", "run-42", not_before=NOW.isoformat())
    path = config.imports_root / "run-42" / "input"
    assert imported.imported_path.startswith(str(path))
    assert imported.pixel_size == (2480, 3508) and imported.dpi == (300.0, 300.0)
    assert imported.page_size_mm == (209.97, 297.01)
    assert (config.scans_dir / "0776_001.png").exists()  # source untouched
    assert service.audit.tail()[-1]["operation"] == "scan_imported"


def test_import_rejects_scans_older_than_start_time(service, config):
    _scan(config.scans_dir, "old.png", when=NOW - timedelta(hours=2))
    with pytest.raises(HardwareError, match="before the supplied start time"):
        service.import_scan("old.png", "run-1", not_before=NOW.isoformat())


@pytest.mark.parametrize("name", ["..\\secret.png", "../secret.png", "sub/x.png", "C:\\Windows\\win.ini", ".."])
def test_import_rejects_path_traversal(service, name):
    with pytest.raises(HardwareError):
        service.import_scan(name, "run-1")


def test_import_rejects_symlink_escape(service, config, tmp_path):
    outside = tmp_path / "outside.png"
    Image.new("L", (10, 10)).save(outside)
    try:
        (config.scans_dir / "link.png").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need developer mode on Windows")
    with pytest.raises(HardwareError, match="outside"):
        service.import_scan("link.png", "run-1")


def test_import_rejects_unsupported_format_and_bad_run_id(service, config):
    _scan(config.scans_dir, "scan.bmp", content=b"BM")
    with pytest.raises(HardwareError, match="PDF, PNG, JPEG or TIFF"):
        service.import_scan("scan.bmp", "run-1")
    _scan(config.scans_dir, "ok.png")
    with pytest.raises(HardwareError, match="run_id"):
        service.import_scan("ok.png", "../escape")


def test_import_refuses_a_file_that_is_still_being_written(config, printer, scanner, clock):
    path = _scan(config.scans_dir, "growing.pdf", content=b"%PDF")
    ticks = iter(range(1000))
    sizes = iter(range(1, 1000))

    def sleep(_seconds):
        path.write_bytes(b"%PDF" + b"x" * next(sizes))

    cfg = HardwareConfig(artifacts_root=config.artifacts_root, scans_dir=config.scans_dir,
                         scan_stability_seconds=1.0, scan_stability_timeout_seconds=3.0)
    service = make_service(cfg, printer, scanner, clock)
    service.inbox = ScanInbox(cfg.scans_dir, cfg.scan_extensions, sleep=sleep, monotonic=lambda: float(next(ticks)))
    with pytest.raises(HardwareError, match="still changing"):
        service.import_scan("growing.pdf", "run-1")


def test_stability_succeeds_once_file_stops_changing(tmp_path):
    path = tmp_path / "s.pdf"
    path.write_bytes(b"%PDF-1.4 complete")
    ticks = iter(float(i) for i in range(100))
    inbox = ScanInbox(tmp_path, frozenset({".pdf"}), sleep=lambda _s: None, monotonic=lambda: next(ticks))
    assert inbox.wait_until_stable(path, quiet_seconds=2.0, timeout_seconds=10.0)


def test_direct_scan_disabled_by_default(service):
    with pytest.raises(HardwareError, match="direct scanning is disabled"):
        service.acquire_scan("run-1")


def test_direct_scan_requires_remote_scanner_online(config, printer, scanner, clock):
    cfg = HardwareConfig(artifacts_root=config.artifacts_root, scans_dir=config.scans_dir, enable_wia_scan=True)
    service = make_service(cfg, printer, scanner, clock)
    with pytest.raises(HardwareError, match="Remote Scanner"):
        service.acquire_scan("run-1")
    scanner.record = WiaRecord(True, "Color Network ScanGear 2", "2.3.7.0", True, 300)
    result = service.acquire_scan("run-1", "color")
    assert result.pixel_size == (2480, 3508) and len(scanner.acquired) == 1
