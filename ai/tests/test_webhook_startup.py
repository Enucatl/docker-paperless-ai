"""Tests for AI startup dependency initialization."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import niquests
import pytest

from paperless_ai.search import webhook


class _Response:
    headers = {"x-version": "3.0.5"}
    status_code = 200

    def raise_for_status(self) -> None:
        return None


class _PaperlessClient:
    def __init__(self, *, api_failures: int = 0, workflow_failures: int = 0) -> None:
        self.api_calls = 0

        async def get(_: str) -> _Response:
            self.api_calls += 1
            if self.api_calls <= api_failures:
                raise niquests.ConnectionError("Paperless is still starting")
            return _Response()

        self._client = SimpleNamespace(get=AsyncMock(side_effect=get))
        self.workflow_failures = workflow_failures
        self.workflow_calls = 0
        self.custom_field_calls: list[str] = []

    async def ensure_ai_workflows(self, **_: object) -> tuple[int, int]:
        self.workflow_calls += 1
        if self.workflow_calls <= self.workflow_failures:
            raise niquests.ConnectionError("Paperless is still starting")
        return 11, 12

    async def get_or_create_custom_field(self, name: str, data_type: str) -> int:
        self.custom_field_calls.append(name)
        return {"ai_processed": 21, "ai_summary": 22, "ai_result": 23}[name]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        manage_paperless_workflows=True,
        tag_ocr="ai:run-ocr",
        paperless_webhook_url="http://listener/webhook/document",
        webhook_secret="secret",
    )


@pytest.mark.asyncio
async def test_initialize_paperless_retries_transient_setup_failure(
    monkeypatch,
) -> None:
    client = _PaperlessClient(workflow_failures=1)
    sleep = AsyncMock()
    monkeypatch.setattr(webhook.asyncio, "sleep", sleep)

    result = await webhook._initialize_paperless(
        client, _config(), retry_delay=0.01, max_retry_delay=0.01
    )

    assert result == (21, 22, 23)
    assert client.workflow_calls == 2
    assert client.custom_field_calls == [
        "ai_processed",
        "ai_summary",
        "ai_result",
    ]
    sleep.assert_awaited_once_with(0.01)


@pytest.mark.asyncio
async def test_initialize_paperless_retries_api_startup_failure(monkeypatch) -> None:
    client = _PaperlessClient(api_failures=1)
    sleep = AsyncMock()
    monkeypatch.setattr(webhook.asyncio, "sleep", sleep)

    result = await webhook._initialize_paperless(
        client, _config(), retry_delay=0.01, max_retry_delay=0.01
    )

    assert result == (21, 22, 23)
    assert client.api_calls == 2
    sleep.assert_awaited_once_with(0.01)


@pytest.mark.asyncio
async def test_initialize_paperless_does_not_retry_authentication_failure(
    monkeypatch,
) -> None:
    response = SimpleNamespace(status_code=401)

    class AuthenticationError(Exception):
        def __init__(self) -> None:
            super().__init__("unauthorized")
            self.response = response

    client = SimpleNamespace(
        _client=SimpleNamespace(get=AsyncMock(side_effect=AuthenticationError()))
    )
    sleep = AsyncMock()
    monkeypatch.setattr(webhook.asyncio, "sleep", sleep)

    with pytest.raises(AuthenticationError):
        await webhook._initialize_paperless(client, _config())

    sleep.assert_not_awaited()
