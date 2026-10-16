"""Printer enumeration, status decoding and queue job parsing with fake adapters."""

from printgpt_hardware_mcp.service import HardwareService
from printgpt_hardware_mcp.windows_printing import (
    JOB_STATUS_FLAGS,
    JobRecord,
    PrinterRecord,
    a4_geometry_ok,
    decode_flags,
    placement_rect,
)

from hw_fakes import A4_GEOMETRY, IP, QUEUE, make_service


def test_lists_printers_and_marks_only_the_allowlisted_queue(service):
    printers = {p.name: p for p in service.list_printers()}
    assert printers[QUEUE].allowlisted and printers[QUEUE].is_default
    assert not printers["Microsoft Print to PDF"].allowlisted


def test_ready_printer_reports_driver_port_and_defaults(service):
    status = service.printer_status()
    assert status.found and status.ready and status.online
    assert status.port_host_address == IP and status.port_matches_allowlist
    assert status.driver_version == "3.50.0.0"
    assert status.default_paper_size == "Letter" and status.default_duplex == "duplex long edge"


def test_offline_and_error_states_are_not_ready(service, printer):
    printer.printers[0] = PrinterRecord(QUEUE, "drv", IP, 0x80 | 0x02, 0, 0)
    status = service.printer_status()
    assert {"offline", "error"} <= set(status.states)
    assert not status.ready and not status.online


def test_work_offline_attribute_and_unreachable_ip(config, printer, scanner, clock):
    printer.printers[0] = PrinterRecord(QUEUE, "drv", IP, 0, 0x400, 0)
    status = make_service(config, printer, scanner, clock, reachable=False).printer_status()
    assert "work_offline" in status.states and "network_unreachable" in status.states
    assert status.network_reachable is False and not status.ready


def test_port_pointing_elsewhere_is_not_ready(service, printer):
    printer.host = "10.1.3.99"
    status = service.printer_status()
    assert not status.port_matches_allowlist and not status.ready


def test_missing_queue_is_reported_not_raised(service, printer):
    printer.printers = printer.printers[1:]
    status = service.printer_status()
    assert not status.found and not status.ready


def test_queue_jobs_are_parsed_and_foreign_jobs_anonymised(service, printer):
    printer.jobs = [JobRecord(7, "Payroll.xlsx", 0x10 | 0x08, pages_printed=1, total_pages=3)]
    jobs = service.list_print_jobs()
    assert jobs[0].job_id == 7 and set(jobs[0].status) == {"printing", "spooling"}
    assert jobs[0].document == "(not created by PrintGPT)" and not jobs[0].created_by_printgpt
    assert decode_flags(0x1000 | 0x80, JOB_STATUS_FLAGS) == ["printed", "complete"]


def test_capabilities_expose_a4_printable_area(service):
    caps = service.printer_capabilities()
    assert not caps.color_supported and caps.duplex_supported
    area = caps.a4_printable_area
    assert area.device_dpi == (600, 600)
    assert area.margins_mm == (5.08, 5.08, 5.08, 5.08)


def test_a4_geometry_and_exact_placement():
    ok, _ = a4_geometry_ok(A4_GEOMETRY)
    assert ok
    assert not a4_geometry_ok({**A4_GEOMETRY, "PHYSICALWIDTH": 5100, "PHYSICALHEIGHT": 6600})[0]  # Letter
    # The whole sheet is covered from paper (0, 0): no centering and no fit-to-printable-area.
    assert placement_rect(A4_GEOMETRY) == (-120, -120, 4840, 6894)


def test_hardware_unavailable_degrades_gracefully(config, clock):
    service = HardwareService(config, printer=None, scanner=None, reachable=lambda _ip: False, now=clock)
    assert not service.printer_status().found
    scanner = service.scanner_status()
    assert not scanner.wia_device_found and scanner.acquisition_mode == "manual_import"
    overview = service.hardware_capabilities()
    assert "error" in overview["printer_capabilities"]
