import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_ai.search.chat_agent import route_tools
from paperless_ai.search.tools import execute_tool_call, get_available_metadata, parse_tool_arguments, search_documents


def test_route_tools_goes_to_tool_node_when_tool_calls_present():
    state = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call_1", "function": {"name": "search_documents", "arguments": "{}"}}],
            }
        ]
    }
    assert route_tools(state) == "tool_node"


def test_route_tools_ends_when_no_tool_calls():
    assert route_tools({"messages": [{"role": "assistant", "content": "done"}]}) == "__end__"


def test_parse_tool_arguments_accepts_json_strings_and_dicts():
    assert parse_tool_arguments({"doc_id": 42}) == {"doc_id": 42}
    assert parse_tool_arguments('{"doc_id": 42}') == {"doc_id": 42}
    assert parse_tool_arguments("") == {}


@pytest.mark.asyncio
async def test_get_available_metadata_formats_lists():
    client = AsyncMock()
    client.get_available_metadata.return_value = {
        "correspondents": ["Acme Corp"],
        "document_types": ["Invoice"],
        "storage_paths": ["Archive/2024"],
        "tags": ["Paid", "Tax"],
    }
    result = await get_available_metadata(client=client)
    assert "Available Correspondents: Acme Corp" in result
    assert "Available Document Types: Invoice" in result
    assert "Available Storage Paths: Archive/2024" in result
    assert "Available Tags: Paid, Tax" in result


@pytest.mark.asyncio
async def test_search_documents_formats_qdrant_hits():
    embedder = AsyncMock()
    embedder.embed_query.return_value = MagicMock(dense=[0.1] * 1024)

    hit = MagicMock()
    hit.payload = {
        "doc_id": 42,
        "title": "Invoice 42",
        "correspondent": "Acme Corp",
        "document_type": "Invoice",
        "storage_path": "Archive/2024",
        "tags": ["Paid"],
        "date": "2024-01-15",
        "text": "Line item one and line item two",
    }

    qdrant = AsyncMock()
    qdrant.query_points.return_value = [hit]

    with patch("paperless_ai.search.tools.AsyncQdrantClient", return_value=qdrant):
        result = await search_documents(
            "invoice",
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            correspondent="Acme Corp",
        )

    assert "Doc 42" in result
    assert "Invoice 42" in result
    assert "Acme Corp" in result
    assert "Type: Invoice" in result
    assert "Tags: Paid" in result
    qdrant.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_documents_handles_tuple_shaped_qdrant_response():
    embedder = AsyncMock()
    embedder.embed_query.return_value = MagicMock(dense=[0.1] * 1024)

    hit = MagicMock()
    hit.payload = {
        "doc_id": 99,
        "title": "Zoo Tickets",
        "correspondent": "Zoo Zurich",
        "date": "2025-07-01",
        "text": "Family admission tickets purchased online",
    }

    qdrant = AsyncMock()
    qdrant.query_points.return_value = ([hit], None)

    with patch("paperless_ai.search.tools.AsyncQdrantClient", return_value=qdrant):
        result = await search_documents(
            "zoo",
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
        )

    assert "Doc 99" in result
    assert "Zoo Tickets" in result
    qdrant.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_tool_call_reads_document():
    client = AsyncMock()
    client.get_document_with_content.return_value = {
        "id": 7,
        "title": "Receipt",
        "content": "Full OCR text",
    }
    embedder = AsyncMock()

    result = await execute_tool_call(
        "read_full_document",
        {"doc_id": 7, "max_chars": 8000},
        client=client,
        embedder=embedder,
        qdrant_url="http://qdrant:6333",
    )

    assert result == "[Doc 7 | Receipt]\nFull OCR text"
