"""Register (or remove) the PrintGPT hardware MCP server in the Codex config.

Usage (from the repository root, with the project venv):

    .venv\\Scripts\\python.exe scripts\\register_codex_mcp.py            # add, printing disabled
    .venv\\Scripts\\python.exe scripts\\register_codex_mcp.py --allow-print
    .venv\\Scripts\\python.exe scripts\\register_codex_mcp.py --remove
    .venv\\Scripts\\python.exe scripts\\register_codex_mcp.py --print-only  # show the TOML, change nothing

Only the ``[mcp_servers.printgpt-hardware]`` tables are added or replaced. A
timestamped backup of ``config.toml`` is written before any change. The server
uses local stdio transport; nothing listens on the network.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import sys
import tomllib

NAME = "printgpt-hardware"
REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"


def _literal(path: Path | str) -> str:
    text = str(path)
    if "'" in text:
        raise SystemExit(f"path contains a quote and cannot be written as a TOML literal: {text}")
    return f"'{text}'"


def block(allow_print: bool, scans_dir: Path) -> str:
    return f"""
[mcp_servers.{NAME}]
command = {_literal(PYTHON)}
args = ["-m", "printgpt_hardware_mcp"]
cwd = {_literal(REPO)}
startup_timeout_sec = 30
tool_timeout_sec = 120
default_tools_approval_mode = "writes"

[mcp_servers.{NAME}.env]
PRINTGPT_HW_ARTIFACTS_ROOT = {_literal(REPO / "artifacts")}
PRINTGPT_HW_SCANS_DIR = {_literal(scans_dir)}
PRINTGPT_HW_PRINTER_QUEUE = "Canon Generic Plus UFR II"
PRINTGPT_HW_PRINTER_IP = "10.1.3.175"
PRINTGPT_HW_ALLOW_PRINT = "{1 if allow_print else 0}"
PRINTGPT_HW_ENABLE_WIA_SCAN = "0"

[mcp_servers.{NAME}.tools.submit_print_job]
approval_mode = "prompt"

[mcp_servers.{NAME}.tools.cancel_print_job]
approval_mode = "prompt"
"""


def strip_existing(text: str) -> str:
    """Drop every table whose header starts with [mcp_servers.printgpt-hardware."""
    out, skipping = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("["):
            header = stripped.strip("[]").strip()
            skipping = header == f"mcp_servers.{NAME}" or header.startswith(f"mcp_servers.{NAME}.")
        if not skipping:
            out.append(line)
    return "".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-print", action="store_true", help="let submit_print_job reach the printer (still needs explicit confirmation)")
    parser.add_argument("--remove", action="store_true", help="remove the server from the Codex config")
    parser.add_argument("--print-only", action="store_true", help="print the TOML block and exit")
    parser.add_argument("--scans-dir", type=Path, default=Path.home() / "OneDrive" / "Documents" / "Scans")
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "config.toml")
    args = parser.parse_args(argv)

    snippet = block(args.allow_print, args.scans_dir.resolve())
    if args.print_only:
        print(snippet)
        return 0
    if not PYTHON.is_file():
        print(f"project venv not found at {PYTHON}; create it and install with: pip install -e .[hardware]", file=sys.stderr)
        return 1
    original = args.config.read_text(encoding="utf-8") if args.config.is_file() else ""
    tomllib.loads(original)  # refuse to touch a config we cannot parse
    updated = strip_existing(original)
    if not args.remove:
        updated += snippet
    tomllib.loads(updated)
    if original:
        backup = args.config.with_name(f"config.toml.bak-printgpt-{datetime.now():%Y%m%d%H%M%S}")
        shutil.copy2(args.config, backup)
        print(f"backup: {backup}")
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(updated, encoding="utf-8")
    print(("removed" if args.remove else "registered") + f" MCP server '{NAME}' in {args.config}")
    if not args.remove:
        print("physical printing: " + ("ENABLED (still requires prepare + explicit confirmation)" if args.allow_print else "disabled (dry runs only)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
