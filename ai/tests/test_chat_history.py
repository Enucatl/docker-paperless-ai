"""Integration coverage for durable copilot conversations."""

import os
import uuid

import pytest

from paperless_ai.search.chat_store import ChatStore


async def test_chat_store_persists_owned_conversations_and_sources():
    """Store completed turns, isolate owners, and cascade deletion."""
    database_url = os.environ.get("CHAT_DATABASE_URL")
    if not database_url:
        pytest.skip("CHAT_DATABASE_URL is required for chat-store integration")
    store = ChatStore(database_url, os.environ.get("CHAT_DATABASE_PASSWORD"))
    await store.migrate()
    owner_id = f"test-{uuid.uuid4()}"
    other_owner_id = f"test-{uuid.uuid4()}"
    conversation = await store.create_conversation(owner_id)
    assert await store.append_turn(
        conversation["id"],
        owner_id,
        "Find my receipt",
        "I found it.",
        [{"name": "search_documents", "summary": "Found 1 matching document."}],
        "test-model",
        {"total_tokens": 3},
        [{"id": 42, "matched": True, "inspected": True, "title": "Receipt"}],
    )
    assert await store.load_conversation(conversation["id"], other_owner_id) is None

    loaded = await store.load_conversation(conversation["id"], owner_id)
    assert loaded is not None
    assert loaded["title"] == "Find my receipt"
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["sources"] == [
        {
            "id": 42,
            "matched": True,
            "inspected": True,
            "relevance": None,
            "title": "Receipt",
        }
    ]

    assert await store.delete_conversation(conversation["id"], owner_id)
    assert await store.load_conversation(conversation["id"], owner_id) is None
