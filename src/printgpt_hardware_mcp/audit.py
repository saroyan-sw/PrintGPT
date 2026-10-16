"""Append-only JSONL audit trail and the ledger of jobs this server created."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
from typing import Any

_ALLOWED_KEYS = {
    "timestamp", "operation", "result", "run_id", "candidate_id", "artifact_sha256", "artifact_path",
    "printer_queue", "requested_settings", "effective_settings", "unconfirmed_settings", "windows_job_id",
    "preparation_id", "dry_run", "error", "document_name", "scan_filename", "imported_path",
}


class AuditLog:
    """One JSON object per line. Only allowlisted keys are written."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def record(self, operation: str, result: str, **fields: Any) -> dict[str, Any]:
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "operation": operation, "result": result}
        entry.update({k: v for k, v in fields.items() if k in _ALLOWED_KEYS and v is not None})
        if "error" in entry:
            entry["error"] = str(entry["error"])[:500]
        line = json.dumps(entry, sort_keys=True, default=str)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        return entry

    def tail(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


class JobLedger:
    """Persistent record of Windows job IDs submitted by this server."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"jobs": []}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"jobs": []}
        return data if isinstance(data.get("jobs"), list) else {"jobs": []}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._load()["jobs"])

    def add(self, entry: dict[str, Any]) -> None:
        with self._lock:
            data = self._load()
            data["jobs"].append(entry)
            self._save(data)

    def update(self, job_id: int, queue: str, **changes: Any) -> None:
        with self._lock:
            data = self._load()
            for job in data["jobs"]:
                if job.get("windows_job_id") == job_id and job.get("queue") == queue:
                    job.update(changes)
            self._save(data)

    def find(self, job_id: int, queue: str) -> dict[str, Any] | None:
        return next((j for j in self.jobs() if j.get("windows_job_id") == job_id and j.get("queue") == queue), None)

    def submitted_hashes(self) -> set[str]:
        return {j["artifact_sha256"] for j in self.jobs() if j.get("state") != "failed" and j.get("artifact_sha256")}
