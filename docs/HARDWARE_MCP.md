# PrintGPT hardware MCP server

A local Python MCP server that gives Codex controlled access to the Canon
imageRUNNER 2425i attached to this Windows laptop. It is an adapter around the
installed Windows drivers and PrintGPT artifacts, not a driver. See
[HARDWARE_AUDIT.md](HARDWARE_AUDIT.md) for the audited queue and scanner facts.

## What it does and does not do

It does:

- report printer queue, driver, port/IP, reachability, state and queued jobs;
- report Canon ScanGear (WIA/TWAIN) availability and whether Remote Scanner is online;
- list scans saved after a given time and import one **explicitly named** scan into a PrintGPT run folder;
- validate an approved toner-only A4 artifact and prepare a print job (no printing);
- submit **one** page, **one** copy, only after preparation and explicit confirmation;
- list and cancel only the jobs it submitted;
- write a JSON audit trail.

It does not: install or replace drivers; change printer, port, network, firmware or admin
settings; open any network listener (stdio only); print scans, previews or arbitrary
files; print more than one copy or batches; or choose a scan for you.

## Requirements

- Windows 11 with the **Canon Generic Plus UFR II** driver (3.50) and a queue named
  exactly `Canon Generic Plus UFR II` on a TCP/IP port to `10.1.3.175`.
- Canon Color Network ScanGear 2 (2.3.7) for scanning, or the printer's Scan & Send.
- The project venv (Python 3.12) with the hardware extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[app,hardware]"
```

This pins `mcp==2.3.0` (official Python SDK; `MCPServer`, stdio transport) and pywin32.

## Configuration

All values have safe defaults and can be set as environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PRINTGPT_HW_PRINTER_QUEUE` | `Canon Generic Plus UFR II` | The only queue the server will use |
| `PRINTGPT_HW_PRINTER_IP` | `10.1.3.175` | The queue's port must resolve to this address |
| `PRINTGPT_HW_ARTIFACTS_ROOT` | `<repo>\artifacts` | Only artifacts below this folder can be validated/printed |
| `PRINTGPT_HW_SCANS_DIR` | `%USERPROFILE%\OneDrive\Documents\Scans` | Only folder scans are imported from |
| `PRINTGPT_HW_ALLOW_PRINT` | `0` | `1` lets `submit_print_job(dry_run=false)` reach the printer |
| `PRINTGPT_HW_ENABLE_WIA_SCAN` | `0` | `1` registers the experimental `acquire_scan` tool |
| `PRINTGPT_HW_MANIFEST_MAX_AGE_HOURS` | `24` | Approvals older than this are refused |

Server state: `artifacts\hardware_mcp\audit.jsonl` (audit trail),
`artifacts\hardware_mcp\jobs.json` (jobs this server created),
`artifacts\hardware_mcp\imports\<run_id>\input\` (imported scans).

## Run it directly

```powershell
.\.venv\Scripts\python.exe -m printgpt_hardware_mcp
```

The process speaks MCP over stdin/stdout and waits for a client. Stop it with Ctrl+C.

## Register with Codex

Codex reads `[mcp_servers.<name>]` tables from `%USERPROFILE%\.codex\config.toml`.
`codex mcp add` cannot set `cwd` or per-tool approval, so use the repo script
(it backs up the config and only edits the `printgpt-hardware` tables):

```powershell
.\.venv\Scripts\python.exe scripts\register_codex_mcp.py
```

Add `--allow-print` to enable physical printing, `--print-only` to see the TOML
without changing anything, and `--remove` to unregister. The block it writes:

```toml
[mcp_servers.printgpt-hardware]
command = 'C:\Users\saroy\Codes\PrintGPT\.venv\Scripts\python.exe'
args = ["-m", "printgpt_hardware_mcp"]
cwd = 'C:\Users\saroy\Codes\PrintGPT'
startup_timeout_sec = 30
tool_timeout_sec = 120
default_tools_approval_mode = "writes"

[mcp_servers.printgpt-hardware.env]
PRINTGPT_HW_ARTIFACTS_ROOT = 'C:\Users\saroy\Codes\PrintGPT\artifacts'
PRINTGPT_HW_SCANS_DIR = 'C:\Users\saroy\OneDrive\Documents\Scans'
PRINTGPT_HW_PRINTER_QUEUE = "Canon Generic Plus UFR II"
PRINTGPT_HW_PRINTER_IP = "10.1.3.175"
PRINTGPT_HW_ALLOW_PRINT = "0"
PRINTGPT_HW_ENABLE_WIA_SCAN = "0"

[mcp_servers.printgpt-hardware.tools.submit_print_job]
approval_mode = "prompt"

[mcp_servers.printgpt-hardware.tools.cancel_print_job]
approval_mode = "prompt"
```

`default_tools_approval_mode = "writes"` makes Codex ask before any tool not
marked read-only (`import_scan`, `submit_print_job`, `cancel_print_job`); submit and
cancel always prompt. Verify:

```powershell
codex mcp list
codex mcp get printgpt-hardware
```

Then start a **new** Codex session (servers load at session start) and ask it to
call `get_printer_status`, `get_scanner_status` and `prepare_print_job` on an
approved artifact. Do not print to test the registration.

## Tools

| Tool | Side effect |
| --- | --- |
| `get_hardware_capabilities` | Read-only |
| `list_printers` | Read-only |
| `get_printer_status` | Read-only (TCP connect to port 9100, no data sent) |
| `list_print_jobs` | Read-only; foreign jobs are anonymised |
| `get_scanner_status` | Read-only (WIA property read, no scan) |
| `find_recent_scans(since, limit)` | Read-only |
| `import_scan(filename, run_id, not_before?)` | Copies one file into `artifacts\hardware_mcp\imports` |
| `validate_print_artifact(artifact_path)` | Read-only |
| `prepare_print_job(artifact_path, copies=1)` | Read-only for the device: records a 15-minute preparation and an audit line; **does not print** |
| `submit_print_job(preparation_id, artifact_sha256, confirm_physical_print, dry_run=true)` | **Prints one page; uses paper and toner** when `dry_run=false` |
| `cancel_print_job(windows_job_id)` | Deletes a PrintGPT job from the Windows queue |
| `acquire_scan(run_id, color_mode)` | Only when `PRINTGPT_HW_ENABLE_WIA_SCAN=1`; **starts the scanner** |

## Scanning

### Supported workflow: manual import

1. Note the current time, then scan one A4 page with ScanGear (or Scan & Send to this
   folder). With ScanGear, select **Scan → Remote Scanner** on the printer panel first.
2. `find_recent_scans(since="<that time, ISO-8601>")` lists only newer PDF/PNG/JPEG/TIFF files.
3. You pick the file. The server never takes the newest file automatically.
4. `import_scan(filename="0776_001.pdf", run_id="demo-1", not_before="<that time>")`
   waits until the file size stops changing, copies it, verifies the copy's SHA-256 and
   returns path, hash, page count, page size and DPI.
5. Load the imported file in the PrintGPT app as scanner input. Imported scans are
   inputs only and can never be printed.

### Direct WIA scanning (experimental, off by default)

ScanGear registers a WIA 2.0 device, so programmatic acquisition is plausible. It only
works after the panel is in Remote Scanner mode, and it has **not** been verified with a
real page. With `PRINTGPT_HW_ENABLE_WIA_SCAN=1` the `acquire_scan` tool sets A4,
300 DPI, colour or greyscale, and runs the transfer in a subprocess with a 180 s
timeout. If it hangs or the device is offline, use manual import.

## Safe printing workflow

1. In PrintGPT, approve a validated candidate. `approve_and_export` writes
   `print\approved_toner_only.png` (2480 × 3508, 300 DPI), `print\approved_a4_actual_size.pdf`
   and `print\print_approval.json` (`printgpt-print-approval/1`: run and candidate IDs,
   approval state and time, artifact SHA-256 values, page geometry, input mode,
   deterministic validation result, simulated flag, toner-only flag, requested settings).
2. `validate_print_artifact` → `prepare_print_job`. Preparation checks:
   - the path resolves (after symlinks and `..`) inside the artifacts root, and not in
     a run `input` folder, the scans folder or the imports folder;
   - the format is PNG or PDF, there is exactly one page, and it is A4 portrait
     (PDF 210 × 297 mm ± 0.5 mm, /Rotate 0; PNG exactly 2480 × 3508);
   - the pixels are pure black/white toner, with no grey scan or preview content, and
     coverage is between 0 % and 60 %;
   - no ink falls inside the driver's 5.08 mm non-printable margin;
   - the sibling manifest has explicit approval newer than 24 h, accepted and not
     simulated validation, the final toner-only flag, and a matching SHA-256;
   - the copy count is 1, the queue is the allowlisted one and the port IP matches;
   - this artifact has not already been submitted.
3. Read the returned summary to the user. Only when the user explicitly says to print,
   call `submit_print_job(preparation_id, artifact_sha256, confirm_physical_print=true,
   dry_run=false)`. The server recomputes the hash and repeats every check first.

How it prints: the 300-DPI binary raster is drawn through GDI onto a printer DC
created with an explicit DEVMODE (A4, portrait, 1 copy, simplex, monochrome, 100 %),
which the driver merges and the server reads back. Before spooling, the server checks
that the DC really describes a 210 × 297 mm sheet. Pixel (0, 0) is placed at the
paper's top-left corner by subtracting the driver's printable-area offset, at exactly
2× scale for 600 DPI, with no fit, shrink, centering or rotation. PDFs are never sent
raw to the UFR II queue. The result reports the Windows job ID and the settings the
driver did and did not confirm. A spooled job is not proof of correct physical
output: measure the first print.

## Troubleshooting

- **Printer offline / not ready**: `get_printer_status` shows `network_unreachable` →
  check the device is on and that 10.1.3.175 is reachable from this network. Windows
  alone shows "Normal" for an off device. `paused`/`error`/`paper_out` → clear it on
  the panel or in *Settings → Printers*.
- **Remote Scanner**: `device_reports_online: false` → on the printer press
  **Scan → Remote Scanner** (or use Scan & Send). ScanGear times out otherwise.
- **TWAIN UI**: ScanGear's TWAIN source (`SGIR.DS`) needs its interactive window. Use it
  manually, save into the scans folder, then import.
- **Stuck job**: `list_print_jobs`, then `cancel_print_job(<id>)` for PrintGPT jobs. Other
  jobs must be cleared from Windows' queue window.
- **Wrong scaling or offset on paper**: the server never fits to page. Check the
  printer's own settings (Canon *Enlarge Print Region*, tray size A4) and measure with
  the calibration page from `docs/PHYSICAL_ALIGNMENT.md`.
- **Unreachable IP**: confirm the laptop is on the printer's network. Do not change the
  port configuration from this tool.
- **Tools missing in Codex**: run `codex mcp list`, check the venv path in the TOML,
  and start a new Codex session.

## Disable or remove

- Disable physical printing only: re-run the script without `--allow-print`.
- Disable the server: set `enabled = false` under `[mcp_servers.printgpt-hardware]`.
- Remove: `.\.venv\Scripts\python.exe scripts\register_codex_mcp.py --remove`.

The server is for this laptop only. Do not publish it, expose it over HTTP or share
its configuration.
