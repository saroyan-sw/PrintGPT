"""Allowlists and safety switches for the hardware MCP server.

Every value can be overridden with a ``PRINTGPT_HW_*`` environment variable so
the Codex registration can pin absolute paths. Physical printing is disabled
unless ``PRINTGPT_HW_ALLOW_PRINT=1``; direct WIA scanning is disabled unless
``PRINTGPT_HW_ENABLE_WIA_SCAN=1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRINTER_QUEUE = "Canon Generic Plus UFR II"
DEFAULT_PRINTER_IP = "10.1.3.175"
DEFAULT_SCANS_DIR = Path.home() / "OneDrive" / "Documents" / "Scans"
SCAN_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"})
ARTIFACT_EXTENSIONS = frozenset({".png", ".pdf"})


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Resolved, immutable server configuration."""

    artifacts_root: Path = REPO_ROOT / "artifacts"
    scans_dir: Path = DEFAULT_SCANS_DIR
    printer_queue: str = DEFAULT_PRINTER_QUEUE
    printer_ip: str = DEFAULT_PRINTER_IP
    allow_physical_print: bool = False
    enable_wia_scan: bool = False
    manifest_max_age_hours: float = 24.0
    preparation_ttl_seconds: float = 900.0
    scan_stability_seconds: float = 2.0
    scan_stability_timeout_seconds: float = 30.0
    scan_timeout_seconds: float = 180.0
    max_scan_bytes: int = 200 * 1024 * 1024
    scan_extensions: frozenset[str] = field(default=SCAN_EXTENSIONS)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts_root", Path(self.artifacts_root).expanduser().resolve())
        object.__setattr__(self, "scans_dir", Path(self.scans_dir).expanduser().resolve())

    @property
    def hardware_root(self) -> Path:
        return self.artifacts_root / "hardware_mcp"

    @property
    def imports_root(self) -> Path:
        return self.hardware_root / "imports"

    @property
    def audit_log_path(self) -> Path:
        return self.hardware_root / "audit.jsonl"

    @property
    def job_ledger_path(self) -> Path:
        return self.hardware_root / "jobs.json"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "HardwareConfig":
        env = os.environ if env is None else env
        return cls(
            artifacts_root=Path(env.get("PRINTGPT_HW_ARTIFACTS_ROOT") or REPO_ROOT / "artifacts"),
            scans_dir=Path(env.get("PRINTGPT_HW_SCANS_DIR") or DEFAULT_SCANS_DIR),
            printer_queue=env.get("PRINTGPT_HW_PRINTER_QUEUE") or DEFAULT_PRINTER_QUEUE,
            printer_ip=env.get("PRINTGPT_HW_PRINTER_IP") or DEFAULT_PRINTER_IP,
            allow_physical_print=_flag(env, "PRINTGPT_HW_ALLOW_PRINT", False),
            enable_wia_scan=_flag(env, "PRINTGPT_HW_ENABLE_WIA_SCAN", False),
            manifest_max_age_hours=float(env.get("PRINTGPT_HW_MANIFEST_MAX_AGE_HOURS") or 24.0),
        )

    def public_summary(self) -> dict[str, object]:
        return {
            "printer_queue": self.printer_queue,
            "printer_ip": self.printer_ip,
            "artifacts_root": str(self.artifacts_root),
            "scans_dir": str(self.scans_dir),
            "imports_root": str(self.imports_root),
            "allow_physical_print": self.allow_physical_print,
            "enable_wia_scan": self.enable_wia_scan,
            "manifest_max_age_hours": self.manifest_max_age_hours,
            "transport": "stdio (local only)",
        }
