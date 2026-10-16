"""PrintGPT hardware MCP server (local stdio transport only).

Run with ``python -m printgpt_hardware_mcp.server``. The server never opens a
network listener; Codex starts it as a subprocess and talks over stdin/stdout.
"""

from __future__ import annotations

import functools
import logging
import sys
from typing import Any, Callable, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from .models import HardwareError
from .service import HardwareService, default_service

F = TypeVar("F", bound=Callable[..., Any])

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
# Records an in-memory preparation and an audit line only; no device or queue change.
PREPARE_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
LOCAL_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
PHYSICAL = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=True)

INSTRUCTIONS = """\
Controlled access to the Canon imageRUNNER 2425i attached to this Windows laptop for PrintGPT.
Rules: never call submit_print_job unless the user has explicitly asked, in this conversation,
to print the specific prepared artifact. Always call prepare_print_job first and show the user
its summary. Never print to test the connection. Scanning is manual: ask the user to scan, then
use find_recent_scans with the time they started and import_scan with the exact filename they pick.
"""


def _safe(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except HardwareError as exc:
            raise ToolError(f"{exc.code}: {exc.message}") from exc
    return wrapper  # type: ignore[return-value]


def build_server(service: HardwareService) -> MCPServer:
    mcp = MCPServer("printgpt-hardware", instructions=INSTRUCTIONS, version="0.1.0")

    @mcp.tool(title="Hardware capabilities", annotations=READ_ONLY)
    @_safe
    def get_hardware_capabilities() -> dict[str, Any]:
        """READ-ONLY. Summarise server configuration, Canon printer status and capabilities,
        scanner status and the supported scan/print workflow. Causes no physical action."""
        return service.hardware_capabilities()

    @mcp.tool(title="List Windows printers", annotations=READ_ONLY)
    @_safe
    def list_printers() -> dict[str, Any]:
        """READ-ONLY. List installed Windows print queues and mark the single allowlisted
        PrintGPT queue. Causes no physical action."""
        return {"printers": [p.model_dump() for p in service.list_printers()]}

    @mcp.tool(title="Canon printer status", annotations=READ_ONLY)
    @_safe
    def get_printer_status() -> dict[str, Any]:
        """READ-ONLY. Report the allowlisted Canon queue: driver, port/IP match, TCP
        reachability, spooler state flags and queued job count. Causes no physical action."""
        return service.printer_status().model_dump()

    @mcp.tool(title="List print jobs", annotations=READ_ONLY)
    @_safe
    def list_print_jobs() -> dict[str, Any]:
        """READ-ONLY. List jobs in the Canon queue. Jobs not created by PrintGPT are shown
        anonymised and cannot be cancelled here. Causes no physical action."""
        return {"jobs": [j.model_dump() for j in service.list_print_jobs()]}

    @mcp.tool(title="Scanner status", annotations=READ_ONLY)
    @_safe
    def get_scanner_status() -> dict[str, Any]:
        """READ-ONLY. Report whether the Canon ScanGear WIA/TWAIN device is visible and
        online, and whether direct scanning or manual import is the active workflow. Does not scan."""
        return service.scanner_status().model_dump()

    @mcp.tool(title="Find recent scans", annotations=READ_ONLY)
    @_safe
    def find_recent_scans(since: str, limit: int = 20) -> dict[str, Any]:
        """READ-ONLY. List PDF/PNG/JPEG/TIFF files in the configured scans folder saved
        after `since` (ISO-8601, e.g. the time the user started scanning). Newest first.
        Does not pick one: ask the user which filename to import. Causes no physical action."""
        return {"since": since, "candidates": [c.model_dump() for c in service.find_recent_scans(since, limit)]}

    @mcp.tool(title="Import a completed scan", annotations=LOCAL_WRITE)
    @_safe
    def import_scan(filename: str, run_id: str, not_before: str | None = None) -> dict[str, Any]:
        """LOCAL FILE COPY, no physical action. Copy one explicitly named scan (bare filename
        from find_recent_scans) from the scans folder into artifacts/hardware_mcp/imports/<run_id>/input/
        after its size is stable. Returns the imported path, SHA-256 and page metadata.
        Imported scans are inputs only and are never printable."""
        return service.import_scan(filename, run_id, not_before).model_dump()

    @mcp.tool(title="Validate print artifact", annotations=READ_ONLY)
    @_safe
    def validate_print_artifact(artifact_path: str) -> dict[str, Any]:
        """READ-ONLY. Check that a file is an approved, toner-only, single-page A4 PrintGPT
        export: allowlisted location, approval manifest, SHA-256, page size/raster size,
        real (non-simulated) validation and printable margins. Causes no physical action."""
        return service.validate_print_artifact(artifact_path).model_dump()

    @mcp.tool(title="Prepare print job (does not print)", annotations=PREPARE_ONLY)
    @_safe
    def prepare_print_job(artifact_path: str, copies: int = 1) -> dict[str, Any]:
        """DOES NOT PRINT. Validate one approved artifact and return the exact queue, file,
        hash, page size, copy count and settings that WOULD be used, plus warnings and a
        preparation_id (valid 15 minutes). Show the summary to the user. Only one copy allowed."""
        return service.prepare_print_job(artifact_path, copies).model_dump()

    @mcp.tool(title="Submit print job — USES PAPER AND TONER", annotations=PHYSICAL)
    @_safe
    def submit_print_job(preparation_id: str, artifact_sha256: str, confirm_physical_print: bool,
                         dry_run: bool = True) -> dict[str, Any]:
        """PHYSICAL ACTION: with dry_run=false this prints one page on the Canon printer and
        USES PAPER AND TONER. Call only after the user has explicitly asked to print this
        prepared artifact. Requires the preparation_id and artifact_sha256 from
        prepare_print_job and confirm_physical_print=true. Defaults to a dry run that sends nothing."""
        return service.submit_print_job(preparation_id, artifact_sha256, confirm_physical_print, dry_run).model_dump()

    @mcp.tool(title="Cancel a PrintGPT print job", annotations=PHYSICAL)
    @_safe
    def cancel_print_job(windows_job_id: int) -> dict[str, Any]:
        """CHANGES THE PRINT QUEUE. Cancel a job that this PrintGPT server submitted.
        Refuses any other job. Pages already transferred to the device may still print."""
        return service.cancel_print_job(windows_job_id).model_dump()

    if service.config.enable_wia_scan:
        @mcp.tool(title="Acquire scan via WIA — STARTS THE SCANNER", annotations=PHYSICAL)
        @_safe
        def acquire_scan(run_id: str, color_mode: str = "grayscale") -> dict[str, Any]:
            """PHYSICAL ACTION (experimental): start one A4, 300-DPI flatbed scan through WIA into
            artifacts/hardware_mcp/imports/<run_id>/input/. The user must have placed the page and
            selected Remote Scanner on the printer panel. color_mode: 'color' or 'grayscale'."""
            return service.acquire_scan(run_id, color_mode).model_dump()

    return mcp


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    build_server(default_service()).run("stdio")


if __name__ == "__main__":
    main()
