"""Local configuration for Gmail scan ingestion.

Addresses and credential paths come from environment variables or an ignored
local ``.env`` file; nothing personal is hard-coded or committed.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
import sys

from pydantic import BaseModel, ConfigDict, Field, field_validator


_ADDRESS = re.compile(r"^[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+$")
_ENV_NAMES = {
    "gmail_account": "PRINTGPT_GMAIL_ACCOUNT",
    "scan_sender": "PRINTGPT_SCAN_SENDER",
    "credentials_path": "PRINTGPT_GMAIL_CREDENTIALS_PATH",
    "token_path": "PRINTGPT_GMAIL_TOKEN_PATH",
    "subject_filter": "PRINTGPT_SCAN_SUBJECT_FILTER",
    "lookback_days": "PRINTGPT_SCAN_LOOKBACK_DAYS",
    "max_attachment_mb": "PRINTGPT_SCAN_MAX_ATTACHMENT_MB",
    "poll_interval_s": "PRINTGPT_SCAN_POLL_INTERVAL_S",
    "timeout_s": "PRINTGPT_SCAN_TIMEOUT_S",
    "state_dir": "PRINTGPT_SCAN_STATE_DIR",
    "runs_root": "PRINTGPT_SCAN_RUNS_ROOT",
}


def application_data_dir() -> Path:
    """Per-user directory for tokens and ingestion state, outside the repository."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "PrintGPT"


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse simple ``KEY=VALUE`` lines; comments, blanks, and quotes are handled."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip().removeprefix("export ").strip()] = value
    return values


class EmailIngestionSettings(BaseModel):
    """Validated ingestion settings. Unset optional values keep safe defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gmail_account: str
    scan_sender: str
    credentials_path: Path | None = None
    token_path: Path = Field(default_factory=lambda: application_data_dir() / "gmail_token.json")
    subject_filter: str | None = None
    lookback_days: int = Field(7, ge=1, le=60)
    max_messages: int = Field(10, ge=1, le=50)
    max_attachment_mb: float = Field(50.0, gt=0, le=200)
    poll_interval_s: float = Field(4.0, ge=2, le=30)
    timeout_s: float = Field(180.0, ge=10, le=3600)
    state_dir: Path = Field(default_factory=lambda: application_data_dir() / "email_ingestion")
    runs_root: Path = Path("artifacts/email_scans")

    @field_validator("gmail_account", "scan_sender")
    @classmethod
    def _valid_address(cls, value: str) -> str:
        value = value.strip().lower()
        if not _ADDRESS.match(value):
            raise ValueError("must be a plain email address")
        return value

    @field_validator("subject_filter")
    @classmethod
    def _blank_subject_is_none(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @property
    def max_attachment_bytes(self) -> int:
        return int(self.max_attachment_mb * 1024 * 1024)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        dotenv_path: Path | None = Path(".env"),
    ) -> EmailIngestionSettings:
        """Read settings from ``.env`` overlaid by real environment variables."""
        values = read_dotenv(dotenv_path) if dotenv_path is not None else {}
        values.update(os.environ if environ is None else environ)
        fields = {
            field: values[name]
            for field, name in _ENV_NAMES.items()
            if values.get(name, "").strip()
        }
        return cls.model_validate(fields)


def missing_settings(environ: Mapping[str, str] | None = None, dotenv_path: Path | None = Path(".env")) -> list[str]:
    """Names of required variables that are not configured."""
    values = read_dotenv(dotenv_path) if dotenv_path is not None else {}
    values.update(os.environ if environ is None else environ)
    required = ("PRINTGPT_GMAIL_ACCOUNT", "PRINTGPT_SCAN_SENDER")
    return [name for name in required if not values.get(name, "").strip()]
