"""Write the approval manifest that a print adapter must verify before printing.

The manifest is a plain JSON contract (``printgpt-print-approval/1``) so that the
hardware MCP server can check it without importing the generation pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader

PRINT_APPROVAL_SCHEMA = "printgpt-print-approval/1"
PRINT_APPROVAL_FILENAME = "print_approval.json"

REQUESTED_PRINT_SETTINGS: dict[str, Any] = {
    "paper_size": "A4",
    "orientation": "portrait",
    "copies": 1,
    "duplex": "simplex",
    "color_mode": "monochrome",
    "scale_percent": 100,
    "fit_to_page": False,
    "pages": 1,
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_facts(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        dpi = image.info.get("dpi")
        return {
            "width_px": image.width,
            "height_px": image.height,
            "mode": image.mode,
            "dpi": None if dpi is None else [round(float(dpi[0]), 3), round(float(dpi[1]), 3)],
        }


def _pdf_facts(path: Path) -> dict[str, Any]:
    reader = PdfReader(str(path))
    page = reader.pages[0]
    box = page.mediabox
    return {
        "page_count": len(reader.pages),
        "page_size_mm": [
            round(float(box.width) * 25.4 / 72.0, 3),
            round(float(box.height) * 25.4 / 72.0, 3),
        ],
        "rotation_degrees": int(page.rotation) % 360,
    }


def write_print_approval(
    *,
    run_dir: Path,
    run_id: str,
    candidate_id: str,
    attempt_id: str,
    toner_png: Path,
    a4_pdf: Path,
    validation: dict[str, Any],
    input_mode: dict[str, Any],
    printer_correction_applied: bool,
    approved_at: datetime | None = None,
) -> Path:
    """Record exactly which toner-only files the user approved, with hashes."""
    approved_at = approved_at or datetime.now(timezone.utc)
    payload = {
        "schema_version": PRINT_APPROVAL_SCHEMA,
        "run_id": run_id,
        "selected_candidate_id": candidate_id,
        "selected_attempt_id": attempt_id,
        "approval": {"state": "approved", "approved_at": approved_at.isoformat()},
        "final_toner_only": True,
        "contains_source_page": False,
        "input_mode": input_mode,
        "validation": {
            "accepted": bool(validation.get("accepted")),
            "simulated": bool(validation.get("simulated")),
            "deterministic": True,
            "reasons": list(validation.get("reasons", [])),
        },
        "printer_correction_applied": printer_correction_applied,
        "artifacts": {
            "toner_png": {
                "path": toner_png.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(toner_png),
                **_png_facts(toner_png),
            },
            "a4_pdf": {
                "path": a4_pdf.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(a4_pdf),
                **_pdf_facts(a4_pdf),
            },
        },
        "requested_print_settings": dict(REQUESTED_PRINT_SETTINGS),
    }
    path = toner_png.parent / PRINT_APPROVAL_FILENAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
