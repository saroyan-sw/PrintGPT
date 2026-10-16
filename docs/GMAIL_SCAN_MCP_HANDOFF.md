# Gmail scan ingestion: hardware MCP hand-off

No hardware MCP branch existed when this work was done, so this branch adds
no MCP server and does not touch printing. The hardware MCP task owns server
registration and printer control. It should expose the tools below as thin
wrappers over `printgpt_mvp.email_ingestion`.

## Shared service

```python
from printgpt_mvp.email_ingestion import connect_email_scan_service, load_settings

connection = connect_email_scan_service(load_settings(), allow_browser=False)
if connection.service is None:
    ...  # report connection.message; consent requires `python -m printgpt_mvp.email_ingestion connect`
service = connection.service  # EmailScanService; create once at server start-up
```

- Never pass `allow_browser=True` from an MCP tool. Consent is a one-time,
  user-attended step.
- Sessions and the processed registry live in the shared state directory
  under a file lock, so the server and the Streamlit app see the same
  sessions and can never import the same email twice.
- All return values are Pydantic models. `model_dump(mode="json")` gives
  JSON-safe tool output.

## Tool mapping

| MCP tool | Python call | Returns |
| --- | --- | --- |
| `start_email_scan_session` | `service.start_scan_session()` | `ScanSession` |
| `get_email_scan_status` | `service.poll_scan_session(session_id)` (checks Gmail once) or `service.get_scan_session_status(session_id)` (no Gmail call) | `PollResult` / `ScanSession` |
| `list_unclaimed_scan_emails` | `[m for m in service.list_matching_scan_messages() if not m.already_processed]` | `tuple[ScanMessage, ...]` |
| `import_scan_email` | `service.import_selected_scan(session_id, message_id, attachment_id)` | `ImportResult` |
| `cancel_email_scan_session` | `service.cancel_scan_session(session_id)` | `ScanSession` |

After `IMPORTED`, call
`service.normalize_imported_scan(session_id, input_mode=InputMode.SCANNER_A4)`.
It returns `(ScanSession, InputNormalizationResult)`, and the session becomes
`awaiting_confirmation`. The user must confirm the preview before generation.

## Models

- `ScanSession`:
  - `session_id`, `started_at` (UTC), `expected_recipient`, `allowed_sender`,
    `status`, `timeout_s`
  - message fields: `message_id`, `attachment_id`, `received_at`, `subject`
  - file fields: `original_filename`, `mime_type`, `size_bytes`, `sha256`,
    `imported_path`, `run_id`
  - `selection_candidates`, `timestamps` (status → time), `warnings`, `error`
- `ScanSessionStatus`: `waiting_for_email`, `needs_selection`,
  `email_found`, `downloading`, `downloaded`, `validating`, `imported`,
  `normalizing`, `awaiting_confirmation`, `failed`, `timed_out`, `cancelled`.
- `PollResult(session, outcome, detail)`, where `outcome` is one of
  `waiting`, `imported`, `needs_selection`, `timed_out`, `cancelled`,
  `failed`, or `not_waiting`.
- `ImportResult(session, outcome, detail)`, where `outcome` is one of
  `imported`, `already_processed`, `rejected`, `not_found`, or
  `session_closed`.
- `ScanMessage(message_id, received_at, subject, attachments, already_processed)`.
  Each attachment is a `ScanAttachment` with `message_id`, `attachment_id`,
  `filename`, `mime_type`, `size_bytes`, and `scan_format`.

Expected failures are result values. Only `UnknownSessionError` (a
`ValueError`) and `StoreLockTimeoutError` are raised.

## Integration steps

1. Merge `feature/gmail-scan-ingestion` and install `.[gmail]`.
2. In the MCP server, build one `EmailScanService` at start-up as shown above.
3. Register the five tools with the mapping in this document. Poll
   `get_email_scan_status` every 3–5 s, using the session's `timeout_s`.
4. Hand the imported file to the existing Scanner A4 normalization with
   `normalize_imported_scan`. Do not route email attachments through ArUco
   detection unless the user chose Marked-page mode.
5. Keep printing behind the existing explicit approval step. Email ingestion
   never prints.
6. Tests: reuse `tests/email_ingestion/email_fakes.py` (`FakeGmail`,
   `make_service`) for MCP tool tests without a real inbox.
