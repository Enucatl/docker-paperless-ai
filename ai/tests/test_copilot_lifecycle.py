"""The parked copilot starts independently of document processing."""

from unittest.mock import AsyncMock, Mock

import pytest

from paperless_ai.search import app


@pytest.mark.asyncio
async def test_chat_lifecycle_requires_only_chat(monkeypatch):
    """Chat startup opens its own history and client without worker resources."""
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless.test")
    monkeypatch.setenv("PAPERLESS_TOKEN", "token")
    monkeypatch.setenv("CHAT_DATABASE_URL", "postgresql://chat/test")
    monkeypatch.setenv("INFERENCE_CHAT_MODEL", "chat-model")
    monkeypatch.delenv("INFERENCE_METADATA_MODEL", raising=False)
    monkeypatch.setattr(app, "setup_telemetry", Mock())
    client = Mock(aclose=AsyncMock())
    monkeypatch.setattr(app, "PaperlessClient", Mock(return_value=client))
    store = Mock(migrate=AsyncMock())
    monkeypatch.setattr(
        "paperless_ai.search.chat_store.ChatStore", Mock(return_value=store)
    )
    async with app.lifespan(app.app):
        assert (await app.health()).status_code == 200
        store.migrate.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert (await app.health()).status_code == 503
