"""Guided loading of the original sheet and printer checks before a physical print.

The printer cannot tell which sheet is in the tray or which way up it lies, so
loading the original is a guided manual step. A paper-loading profile records
the tray, the face (up or down) and the leading edge. It stays *unverified*
until one print has been checked on paper; while unverified, the app asks for a
spare test sheet instead of the original. Printer checks cover everything the
Windows queue and the network can actually report.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Final

from printgpt_mvp.email_ingestion.settings import application_data_dir


PAPER_PROFILE_ENV: Final[str] = "PRINTGPT_PAPER_PROFILE"
PAPER_PROFILE_SCHEMA: Final[str] = "printgpt-paper-loading/1"
FACES: Final[tuple[str, ...]] = ("down", "up")
LEADING_EDGES: Final[tuple[str, ...]] = ("top", "bottom")
# Windows queue states that mean the paper path needs attention before printing.
# ``manual_feed`` is not one: it is what a printer reports while waiting for the bypass sheet.
PAPER_PATH_STATES: Final[frozenset[str]] = frozenset(
    {"paper_jam", "paper_out", "paper_problem", "door_open", "user_intervention"}
)


@dataclass(frozen=True, slots=True)
class PaperLoadingProfile:
    """How the original sheet goes into the printer so new ink lands on the drawn side, upright."""

    tray: str = "Multi-purpose (bypass) tray"
    face: str = "down"
    leading_edge: str = "top"
    verified: bool = False
    verified_at: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.tray.strip():
            raise ValueError("tray must be named")
        if self.face not in FACES:
            raise ValueError(f"face must be one of {FACES}")
        if self.leading_edge not in LEADING_EDGES:
            raise ValueError(f"leading_edge must be one of {LEADING_EDGES}")

    def instructions(self) -> list[str]:
        side = "drawing side facing down" if self.face == "down" else "drawing side facing up"
        return [
            "Take the original sheet you scanned. Make sure it is flat and nothing else is in the tray.",
            f"Put it in the {self.tray}, {side}.",
            f"Feed the {self.leading_edge} edge of the drawing in first.",
            "Push the paper guides against the sheet without bending it.",
        ]

    def confirmation(self) -> str:
        side = "drawing side down" if self.face == "down" else "drawing side up"
        return (
            f"I placed the original sheet in the {self.tray}, {side}, {self.leading_edge} edge first, "
            "and it is the only sheet in that tray."
        )

    def to_json(self) -> str:
        return json.dumps({"schema": PAPER_PROFILE_SCHEMA, **asdict(self)}, indent=2) + "\n"


def paper_profile_path() -> Path:
    configured = os.environ.get(PAPER_PROFILE_ENV, "").strip()
    return Path(configured) if configured else application_data_dir() / "paper_loading.json"


def load_paper_profile(path: Path | None = None) -> PaperLoadingProfile:
    """The saved profile, or an unverified default when none is saved yet."""
    path = path or paper_profile_path()
    if not path.is_file():
        return PaperLoadingProfile()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.pop("schema", None) != PAPER_PROFILE_SCHEMA:
        raise ValueError(f"{path} is not a {PAPER_PROFILE_SCHEMA} file")
    return PaperLoadingProfile(**data)


def save_paper_profile(profile: PaperLoadingProfile, path: Path | None = None) -> Path:
    path = path or paper_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.to_json(), encoding="utf-8")
    return path


def with_loading(profile: PaperLoadingProfile, *, tray: str, face: str, leading_edge: str) -> PaperLoadingProfile:
    """Changing how the sheet is loaded always clears verification."""
    changed = (tray, face, leading_edge) != (profile.tray, profile.face, profile.leading_edge)
    if not changed:
        return profile
    return replace(profile, tray=tray, face=face, leading_edge=leading_edge, verified=False, verified_at=None)


def mark_verified(profile: PaperLoadingProfile, when: datetime) -> PaperLoadingProfile:
    return replace(profile, verified=True, verified_at=when.isoformat(timespec="seconds"))


@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class PrintReadiness:
    checks: tuple[ReadinessCheck, ...]

    @property
    def ready(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed(self) -> tuple[ReadinessCheck, ...]:
        return tuple(check for check in self.checks if not check.passed)


def check_print_readiness(printer_status: Mapping[str, Any], *, physical_print_enabled: bool) -> PrintReadiness:
    """Turn the printer status into named pass/fail checks shown before printing."""
    states = set(printer_status.get("states") or ())
    paper_problems = sorted(states & PAPER_PATH_STATES)
    queued = int(printer_status.get("queued_jobs") or 0)
    reachable = printer_status.get("network_reachable")
    checks = (
        ReadinessCheck(
            "Printer queue found",
            bool(printer_status.get("found")) and bool(printer_status.get("allowlisted")),
            f"queue {printer_status.get('queue')!r}",
        ),
        ReadinessCheck(
            "Printer reachable on the network",
            reachable is True,
            "answers on the network" if reachable else "no answer: check the printer is on and on this network",
        ),
        ReadinessCheck(
            "Printer ready",
            bool(printer_status.get("ready")),
            "states: " + (", ".join(sorted(states)) or "idle"),
        ),
        ReadinessCheck(
            "Paper path clear",
            not paper_problems,
            "no jam, empty tray or open door reported" if not paper_problems
            else "clear on the printer panel: " + ", ".join(paper_problems),
        ),
        ReadinessCheck(
            "No other jobs waiting",
            queued == 0,
            "queue is empty" if queued == 0 else f"{queued} job(s) would print first; clear the queue",
        ),
        ReadinessCheck(
            "Physical printing enabled",
            physical_print_enabled,
            "enabled" if physical_print_enabled
            else "disabled: start the app with PRINTGPT_HW_ALLOW_PRINT=1 to allow printing",
        ),
    )
    return PrintReadiness(checks)
