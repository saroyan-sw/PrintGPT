"""The MCP surface: tool list, side-effect labels, structured results and errors."""

import anyio

from mcp import Client

from printgpt_hardware_mcp.config import HardwareConfig
from printgpt_hardware_mcp.server import build_server

from hw_fakes import make_service


def _run(service, func):
    async def main():
        async with Client(build_server(service)) as client:
            return await func(client)
    return anyio.run(main)


def test_tools_are_listed_with_side_effect_annotations(service):
    tools = {tool.name: tool for tool in _run(service, lambda c: c.list_tools()).tools}
    read_only = {"get_hardware_capabilities", "list_printers", "get_printer_status", "list_print_jobs",
                 "get_scanner_status", "find_recent_scans", "validate_print_artifact"}
    assert read_only | {"import_scan", "prepare_print_job", "submit_print_job", "cancel_print_job"} == set(tools)
    for name in read_only:
        assert tools[name].annotations.read_only_hint is True, name
        assert "READ-ONLY" in tools[name].description
    submit = tools["submit_print_job"]
    assert submit.annotations.destructive_hint is True and not submit.annotations.read_only_hint
    assert "USES PAPER AND TONER" in submit.description
    assert "DOES NOT PRINT" in tools["prepare_print_job"].description
    assert "acquire_scan" not in tools


def test_acquire_scan_only_registered_when_enabled(config, printer, scanner, clock):
    cfg = HardwareConfig(artifacts_root=config.artifacts_root, scans_dir=config.scans_dir, enable_wia_scan=True)
    tools = {t.name for t in _run(make_service(cfg, printer, scanner, clock), lambda c: c.list_tools()).tools}
    assert "acquire_scan" in tools


def test_status_and_prepare_return_structured_content(service, printer, export):
    async def calls(client):
        status = await client.call_tool("get_printer_status", {})
        prepared = await client.call_tool("prepare_print_job", {"artifact_path": str(export["png"])})
        return status, prepared

    status, prepared = _run(service, calls)
    assert not status.is_error and status.structured_content["ready"] is True
    assert not prepared.is_error and prepared.structured_content["copies"] == 1
    assert printer.submitted == []


def test_hardware_errors_become_tool_errors(service, export):
    async def call(client):
        return await client.call_tool("prepare_print_job", {"artifact_path": str(export["png"]), "copies": 3})

    result = _run(service, call)
    assert result.is_error
    assert "copies_rejected" in result.content[0].text
