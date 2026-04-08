import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_ai.search.chat_agent import ChatCopilot, route_tools
from paperless_ai.search.tools import (
    ToolExecutionResult,
    ToolSourceRef,
    execute_tool_call,
    execute_tool_call_detailed,
    get_available_metadata,
    parse_tool_arguments,
    search_documents,
)


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
    client = AsyncMock()

    point = MagicMock()
    point.payload = {
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
    qdrant.scroll.return_value = ([point], None)

    with (
        patch("paperless_ai.search.tools.AsyncQdrantClient", return_value=qdrant),
        patch("paperless_ai.search.tools.hybrid_retrieve", AsyncMock(return_value=([42], {42: "Line item one and line item two"}))),
    ):
        result = await search_documents(
            "invoice",
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            client=client,
            correspondent="Acme Corp",
        )

    assert "Doc 42" in result
    assert "Invoice 42" in result
    assert "Acme Corp" in result
    assert "Type: Invoice" in result
    assert "Tags: Paid" in result
    qdrant.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_documents_uses_chunk_map_when_scroll_payload_missing():
    embedder = AsyncMock()
    client = AsyncMock()

    qdrant = AsyncMock()
    qdrant.scroll.return_value = ([], None)

    with (
        patch("paperless_ai.search.tools.AsyncQdrantClient", return_value=qdrant),
        patch(
            "paperless_ai.search.tools.hybrid_retrieve",
            AsyncMock(return_value=([99], {99: "Family admission tickets purchased online"})),
        ),
    ):
        result = await search_documents(
            "zoo",
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            client=client,
        )

    assert "Doc 99" in result
    assert "Family admission tickets purchased online" in result
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


@pytest.mark.asyncio
async def test_execute_tool_call_detailed_collects_source_refs():
    client = AsyncMock()
    client.get_document_with_content.return_value = {
        "id": 7,
        "title": "Receipt",
        "content": "Full OCR text",
    }
    embedder = AsyncMock()

    result = await execute_tool_call_detailed(
        "read_full_document",
        {"doc_id": 7, "max_chars": 8000},
        client=client,
        embedder=embedder,
        qdrant_url="http://qdrant:6333",
    )

    assert result.summary == "Read OCR text for document 7."
    assert result.source_refs == [ToolSourceRef(doc_id=7, source_type="read")]


@pytest.mark.asyncio
async def test_chat_copilot_run_turn_emits_events_and_aggregates_usage():
    config = MagicMock()
    config.effective_metadata_model = "openai/test-model"
    config.metadata_api_base = None
    config.get_metadata_litellm_kwargs.return_value = {}

    copilot = ChatCopilot(
        config=config,
        client=AsyncMock(),
        embedder=AsyncMock(),
        qdrant_url="http://qdrant:6333",
    )

    first_response = MagicMock()
    first_response.choices = [
        MagicMock(
            message={
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_documents",
                            "arguments": '{"query":"invoice"}',
                        },
                    }
                ],
            }
        )
    ]
    first_response.usage = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}

    second_response = MagicMock()
    second_response.choices = [
        MagicMock(message={"role": "assistant", "content": "Answer with Doc 42 cited."})
    ]
    second_response.usage = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}

    events = []

    async def capture(event):
        events.append(event)

    with (
        patch("paperless_ai.search.chat_agent.litellm.acompletion", side_effect=[first_response, second_response]),
        patch(
            "paperless_ai.search.chat_agent.execute_tool_call_detailed",
            AsyncMock(
                return_value=ToolExecutionResult(
                    content="[Doc 42 | Invoice 42]\nExcerpt",
                    summary="Found 1 matching document(s).",
                    preview="Doc 42 matched.",
                    source_refs=[ToolSourceRef(doc_id=42, source_type="search")],
                )
            ),
        ),
    ):
        result = await copilot.run_turn("Find invoice 42", event_callback=capture)

    assert result.reply == "Answer with Doc 42 cited."
    assert result.sources == {42: {"matched": True, "inspected": False}}
    assert result.usage == {"prompt_tokens": 30, "completion_tokens": 6, "total_tokens": 36}
    assert any(event["type"] == "tool_call_started" for event in events)
    assert any(event["type"] == "tool_call_completed" for event in events)
    assert any(event["type"] == "usage" and event["scope"] == "step" for event in events)
