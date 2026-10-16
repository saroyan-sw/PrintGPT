"""Opt-in check against the real configured inbox (read-only).

Skipped unless PRINTGPT_LIVE_GMAIL=1 and a token from
``python -m printgpt_mvp.email_ingestion connect`` exists. It only searches
for scanner messages; it imports nothing and opens no message bodies.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PRINTGPT_LIVE_GMAIL") != "1", reason="live Gmail test needs PRINTGPT_LIVE_GMAIL=1"
)


def test_live_connection_and_scanner_search() -> None:
    from printgpt_mvp.email_ingestion.factory import connect_email_scan_service, load_settings

    connection = connect_email_scan_service(load_settings(), allow_browser=False)
    assert connection.connected, connection.message
    messages = connection.service.list_matching_scan_messages()
    assert all(message.attachments for message in messages)
