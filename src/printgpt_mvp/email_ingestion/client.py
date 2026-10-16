"""Minimal read-only Gmail API surface used by ingestion.

``GmailApi`` is the seam for tests: the fake in the test suite implements it
with fixture messages, so no test touches a real inbox.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any, Protocol


class GmailApiError(Exception):
    """A Gmail API request failed; the message never includes mail content."""


class GmailApi(Protocol):
    def profile_email(self) -> str: ...

    def list_message_ids(self, query: str, max_results: int) -> list[str]: ...

    def get_message(self, message_id: str) -> Mapping[str, Any]: ...

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes: ...


def decode_base64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


class GoogleGmailApi:
    """``GmailApi`` backed by ``google-api-python-client`` with read-only credentials."""

    def __init__(self, credentials: Any, *, timeout_s: float = 30.0) -> None:
        import google_auth_httplib2
        import httplib2
        from googleapiclient.discovery import build

        http = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=timeout_s))
        self._service = build("gmail", "v1", http=http, cache_discovery=False)

    def _execute(self, request: Any, action: str) -> Any:
        from googleapiclient.errors import HttpError

        try:
            return request.execute(num_retries=1)
        except HttpError as error:
            raise GmailApiError(f"Gmail {action} failed with HTTP {error.resp.status}") from error
        except (OSError, TimeoutError) as error:
            raise GmailApiError(f"Gmail {action} failed: {type(error).__name__}") from error

    def profile_email(self) -> str:
        profile = self._execute(self._service.users().getProfile(userId="me"), "profile lookup")
        return str(profile.get("emailAddress", "")).lower()

    def list_message_ids(self, query: str, max_results: int) -> list[str]:
        request = self._service.users().messages().list(userId="me", q=query, maxResults=max_results)
        response = self._execute(request, "message search")
        return [str(item["id"]) for item in response.get("messages", [])]

    def get_message(self, message_id: str) -> Mapping[str, Any]:
        # "full" is the only format that lists attachment parts. Text bodies it
        # returns are never read, stored, or logged by ingestion.
        request = self._service.users().messages().get(userId="me", id=message_id, format="full")
        return self._execute(request, "message lookup")

    def get_attachment(self, message_id: str, attachment_id: str) -> bytes:
        request = self._service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=attachment_id
        )
        response = self._execute(request, "attachment download")
        return decode_base64url(str(response.get("data", "")))
