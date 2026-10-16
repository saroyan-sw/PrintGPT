"""Persistent session state and the processed-attachment registry.

State lives in small JSON files under the per-user application-data directory
so a Streamlit rerun, a new browser tab, or the hardware MCP server sees the
same sessions. Writes are atomic (temporary file + rename) and every
read-modify-write runs under a cross-process lock file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import tempfile
import time

from pydantic import BaseModel, ConfigDict

from .models import ProcessedOutcome, ProcessedRecord, ScanSession


_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class StoreLockTimeoutError(Exception):
    """Another process held the ingestion lock for too long."""


class _Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[ProcessedRecord] = []


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class IngestionStore:
    """Sessions in ``sessions/<id>.json`` and processed records in ``processed.json``."""

    def __init__(
        self,
        state_dir: Path,
        *,
        lock_timeout_s: float = 10.0,
        stale_lock_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._root = Path(state_dir)
        self._lock_path = self._root / "ingestion.lock"
        self._lock_timeout_s = lock_timeout_s
        self._stale_lock_s = stale_lock_s
        self._sleep = sleep

    @property
    def root(self) -> Path:
        return self._root

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive lock shared by every process using this state directory."""
        self._root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._lock_timeout_s
        while True:
            try:
                descriptor = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._lock_is_stale():
                    self._lock_path.unlink(missing_ok=True)
                    continue
                if time.monotonic() >= deadline:
                    raise StoreLockTimeoutError(f"could not lock {self._lock_path}") from None
                self._sleep(0.05)
                continue
            break
        try:
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            os.close(descriptor)
            yield
        finally:
            self._lock_path.unlink(missing_ok=True)

    def _lock_is_stale(self) -> bool:
        # A crashed process must not block ingestion forever.
        try:
            age = time.time() - self._lock_path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > self._stale_lock_s

    def _session_path(self, session_id: str) -> Path:
        if not _SESSION_ID.match(session_id):
            raise ValueError(f"invalid session id {session_id!r}")
        return self._root / "sessions" / f"{session_id}.json"

    def save_session(self, session: ScanSession) -> None:
        write_text_atomically(self._session_path(session.session_id), session.model_dump_json(indent=2))

    def load_session(self, session_id: str) -> ScanSession | None:
        path = self._session_path(session_id)
        if not path.is_file():
            return None
        return ScanSession.model_validate_json(path.read_text(encoding="utf-8"))

    def _registry_path(self) -> Path:
        return self._root / "processed.json"

    def processed_records(self) -> tuple[ProcessedRecord, ...]:
        path = self._registry_path()
        if not path.is_file():
            return ()
        return tuple(_Registry.model_validate_json(path.read_text(encoding="utf-8")).records)

    def find_processed(self, *, message_id: str | None = None, sha256: str | None = None) -> ProcessedRecord | None:
        for record in self.processed_records():
            if message_id is not None and record.message_id == message_id:
                return record
            if sha256 is not None and record.sha256 == sha256 and record.outcome is ProcessedOutcome.IMPORTED:
                return record
        return None

    def processed_message_ids(self) -> frozenset[str]:
        return frozenset(record.message_id for record in self.processed_records())

    def add_processed(self, record: ProcessedRecord) -> None:
        """Append a record; call only while holding ``locked()``."""
        registry = _Registry(records=[*self.processed_records(), record])
        write_text_atomically(self._registry_path(), registry.model_dump_json(indent=2))


def utc_now() -> datetime:
    return datetime.now(UTC)
