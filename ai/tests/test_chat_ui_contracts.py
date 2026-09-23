"""Focused tests for chat transport validation and source refresh boundaries."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import niquests
import pytest
from fastapi import HTTPException, WebSocketDisconnect

from paperless_ai.search import webhook


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [
        webhook.load_conversation,
        webhook.rename_conversation,
        webhook.delete_conversation,
    ],
)
async def test_rest_rejects_malformed_conversation_id_before_store(
    monkeypatch, handler
):
    store = SimpleNamespace(
        load_conversation=AsyncMock(),
        rename_conversation=AsyncMock(),
        delete_conversation=AsyncMock(),
    )
    monkeypatch.setattr(webhook, "_chat_store", store)
    request = SimpleNamespace(headers={})

    with pytest.raises(HTTPException) as error:
        if handler is webhook.rename_conversation:
            await handler("bad-id", request, {"title": "Rename"})
        else:
            await handler("bad-id", request)

    assert error.value.status_code == 422
    assert not any(method.await_count for method in vars(store).values())


def test_chat_title_keeps_post_default_and_requires_patch_title():
    assert webhook._chat_title({}) is None
    assert webhook._chat_title({"title": None}) is None
    for value in (None, "", " ", 123, "x" * 121):
        with pytest.raises(HTTPException) as error:
            webhook._chat_title({"title": value}, required=True)
        assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_websocket_rejects_malformed_id_without_store_or_turn(monkeypatch):
    class Socket:
        headers = {}

        def __init__(self):
            self.incoming = iter(['{"content":"hello","conversation_id":"bad-id"}'])
            self.sent = []

        async def accept(self):
            pass

        async def receive_text(self):
            try:
                return next(self.incoming)
            except StopIteration as exc:
                raise WebSocketDisconnect() from exc

        async def send_json(self, payload):
            self.sent.append(payload)

    store = SimpleNamespace(
        load_conversation=AsyncMock(),
        create_conversation=AsyncMock(),
        append_turn=AsyncMock(),
    )
    monkeypatch.setattr(webhook, "_get_chat_copilot", lambda: object())
    monkeypatch.setattr(webhook, "_require_chat_store", lambda: store)
    socket = Socket()

    await webhook.chat_ws(socket)

    assert socket.sent == [
        {"type": "error", "turn_id": "invalid", "content": "Invalid conversation ID."}
    ]
    store.load_conversation.assert_not_awaited()
    store.append_turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_build_and_restore_keep_expected_failures_and_successful_peers(
    monkeypatch,
):
    class Client:
        async def get_document_chat_metadata(self, doc_id):
            if doc_id == 1:
                response = SimpleNamespace(status_code=403)
                raise niquests.HTTPError("forbidden", response=response)
            return {"id": doc_id, "title": "Available", "added": None}

    monkeypatch.setattr(webhook, "_paperless_client", Client())
    sources = await webhook._build_chat_sources(
        {1: {"matched": True}, 2: {"inspected": True}}
    )
    assert sources[0]["id"] == 2 and sources[0]["available"] is True
    assert sources[1]["id"] == 1 and sources[1]["available"] is False
    assert (
        sources[1]["matched"] is True
        and sources[1]["detail_url"] == "/documents/1/detail"
    )

    restored = await webhook._restore_chat_sources(
        [{"id": 1, "title": "Historical title", "matched": True}]
    )
    assert restored[0]["title"] == "Historical title"
    assert restored[0]["available"] is False


@pytest.mark.asyncio
async def test_source_programming_errors_propagate(monkeypatch):
    client = SimpleNamespace(
        get_document_chat_metadata=AsyncMock(side_effect=ValueError("bug"))
    )
    monkeypatch.setattr(webhook, "_paperless_client", client)

    with pytest.raises(ValueError, match="bug"):
        await webhook._build_chat_sources({1: {"matched": True}})
