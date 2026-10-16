"""Command-line checks for Gmail scan ingestion (real inbox, read-only).

    python -m printgpt_mvp.email_ingestion connect   # one-time Google consent
    python -m printgpt_mvp.email_ingestion wait      # wait for one scan, import and normalize it
    python -m printgpt_mvp.email_ingestion revoke    # delete the local token

``wait`` stops after the normalized A4 preview is written; it never runs
generation or printing.
"""

from __future__ import annotations

from argparse import ArgumentParser
import logging
from pathlib import Path
import time

import cv2
from pydantic import ValidationError

from .auth import revoke_local_token
from .factory import connect_email_scan_service, load_settings
from .models import PollOutcome
from .settings import missing_settings


def _connect(allow_browser: bool) -> int:
    connection = connect_email_scan_service(load_settings(), allow_browser=allow_browser)
    print(connection.message)
    return 0 if connection.connected else 1


def _wait() -> int:
    settings = load_settings()
    connection = connect_email_scan_service(settings, allow_browser=False)
    if connection.service is None:
        print(connection.message)
        return 1
    service = connection.service
    session = service.start_scan_session()
    print(f"Session {session.session_id}: send one scan with Canon Scan and Send now.")
    while True:
        result = service.poll_scan_session(session.session_id)
        if result.outcome is PollOutcome.WAITING:
            time.sleep(settings.poll_interval_s)
            continue
        if result.outcome is PollOutcome.NEEDS_SELECTION:
            for item in result.session.selection_candidates:
                print(f"  candidate: {item.message_id} part {item.part_id} {item.filename}")
            print("Several attachments arrived; import one from the app.")
            return 2
        if result.outcome is not PollOutcome.IMPORTED:
            print(f"{result.outcome.value}: {result.detail}")
            return 1
        break
    imported = result.session
    print(f"Imported {imported.original_filename} ({imported.mime_type}, {imported.size_bytes} bytes)")
    print(f"  subject observed: {imported.subject!r}")
    print(f"  sha256: {imported.sha256}")
    print(f"  saved to: {imported.imported_path}")
    _, normalization = service.normalize_imported_scan(session.session_id)
    preview = Path(imported.imported_path).parents[1] / "canonical_a4_preview.png"
    cv2.imwrite(str(preview), cv2.cvtColor(normalization.canonical_rgb, cv2.COLOR_RGB2BGR))
    print(f"  normalized A4 preview: {preview}")
    duplicate = service.import_selected_scan(service.start_scan_session().session_id, imported.message_id or "", imported.part_id or "")
    print(f"  second import of the same message: {duplicate.outcome.value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["connect", "status", "wait", "revoke"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        load_settings()
    except ValidationError as error:
        missing = missing_settings()
        print(f"Missing settings: {', '.join(missing)}" if missing else f"Invalid settings: {error}")
        print("Copy .env.example to .env and fill in the values (see docs/GMAIL_SCAN_INGESTION.md).")
        return 1
    if args.command == "connect":
        return _connect(allow_browser=True)
    if args.command == "status":
        return _connect(allow_browser=False)
    if args.command == "wait":
        return _wait()
    removed = revoke_local_token(load_settings().token_path)
    print("Local token deleted." if removed else "No local token found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
