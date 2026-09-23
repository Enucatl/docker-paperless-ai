from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_common.telemetry import start_span
from paperless_ai.search.chat_agent import ChatCopilot
from paperless_ai.search.tools import (
    TOOL_SCHEMAS,
    ToolExecutionResult,
    ToolSourceRef,
    execute_tool_call,
    execute_tool_call_detailed,
    get_available_metadata,
    parse_tool_arguments,
    search_documents,
)
from shared_inference import CompletionResult, Usage


def _completion(content: str, tool_calls: list[dict] | None = None) -> CompletionResult:
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return CompletionResult(
        content=content,
        message=message,
        tool_calls=tool_calls or [],
        reasoning=None,
        usage=Usage(),
        request_id=None,
        raw={},
    )


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
    assert "Available Correspondents: Acme Corp" in result.content
    assert "Available Document Types: Invoice" in result.content
    assert "Available Storage Paths: Archive/2024" in result.content
    assert "Available Tags: Paid, Tax" in result.content


@pytest.mark.asyncio
async def test_search_documents_returns_paperless_keyword_matches():
    client = AsyncMock()
    client.search_documents_all.return_value = [2231, 1104]
    client.get_document_with_content.side_effect = [
        {"id": 2231, "title": "Zoo ticket", "content": "Zoo admission CHF 30"},
        {"id": 1104, "title": "Zoo visit", "content": "Zoo ticket 2026"},
    ]

    result = await search_documents("zoo", client=client, year="2026", limit=20)

    client.search_documents_all.assert_awaited_once_with(
        "zoo",
        limit=20,
        correspondent=None,
        document_type=None,
        storage_path=None,
        tags=None,
        year="2026",
    )
    assert "Doc 2231" in result.content
    assert "Doc 1104" in result.content
    assert [ref.doc_id for ref in result.source_refs] == [2231, 1104]


async def test_execute_tool_call_reads_document():
    client = AsyncMock()
    client.get_document_with_content.return_value = {
        "id": 7,
        "title": "Receipt",
        "content": "Full OCR text",
    }
    result = await execute_tool_call(
        "read_full_document",
        {"doc_id": 7, "max_chars": 8000},
        client=client,
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
    result = await execute_tool_call_detailed(
        "read_full_document",
        {"doc_id": 7, "max_chars": 8000},
        client=client,
    )

    assert result.summary == "Read OCR text for document 7."
    assert result.source_refs == [ToolSourceRef(doc_id=7, source_type="read")]


def test_search_tool_schema_exposes_mode_enum():
    schema = next(
        tool["function"]
        for tool in TOOL_SCHEMAS
        if tool["function"]["name"] == "search_documents"
    )
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["precision", "recall"]
    assert (
        "short, distinctive keyword query" in schema["description"]
        and "explicit limit" in schema["description"]
    )


@pytest.mark.asyncio
async def test_search_documents_recall_requires_explicit_limit():
    result = await search_documents(
        "youtube premium",
        client=AsyncMock(),
        mode="recall",
    )

    assert result.summary == "Recall searches require an explicit limit."


@pytest.mark.asyncio
async def test_search_documents_rejects_invalid_mode_without_retrieval():
    result = await search_documents(
        "youtube premium",
        client=AsyncMock(),
        mode="recall,precision",
        limit=20,
    )

    assert (
        result.summary
        == "Invalid search mode 'recall,precision'. Allowed values: precision, recall."
    )


@pytest.mark.asyncio
async def test_execute_tool_call_detailed_recall_requires_explicit_limit():
    result = await execute_tool_call_detailed(
        "search_documents",
        {"query": "youtube premium", "mode": "recall"},
        client=AsyncMock(),
    )

    assert result.content == "Recall searches require an explicit limit."


@pytest.mark.asyncio
async def test_execute_tool_call_detailed_rejects_invalid_mode():
    result = await execute_tool_call_detailed(
        "search_documents",
        {"query": "youtube premium", "mode": "recall,precision"},
        client=AsyncMock(),
    )

    assert (
        result.content
        == "Invalid search mode 'recall,precision'. Allowed values: precision, recall."
    )


def test_start_span_preserves_original_exception():
    with pytest.raises(ValueError, match="boom"):
        with start_span("paperless_ai.test.span"):
            raise ValueError("boom")


@pytest.mark.asyncio
async def test_chat_copilot_run_turn_emits_events_and_aggregates_usage():
    config = MagicMock()
    config.chat_model = "openai/chat-model"
    config.metadata_model = "openai/metadata-model"
    config.chat_endpoint = None
    config.metadata_endpoint = None
    config.get_chat_kwargs.return_value = {}

    copilot = ChatCopilot(
        config=config,
        client=AsyncMock(),
    )

    first_response = _completion(
        "",
        [
            {
                "id": "call_1",
                "function": {
                    "name": "search_documents",
                    "arguments": '{"query":"invoice"}',
                },
            }
        ],
    )
    first_response.usage = Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    second_response = _completion("Answer with Doc 42 cited.")
    second_response.usage = Usage(
        prompt_tokens=20, completion_tokens=4, total_tokens=24
    )

    events = []

    async def capture(event):
        events.append(event)

    with (
        patch(
            "paperless_ai.search.chat_agent.complete",
            side_effect=[first_response, second_response],
        ),
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
    assert result.usage == {
        "prompt_tokens": 30,
        "completion_tokens": 6,
        "total_tokens": 36,
    }
    activity = result.tool_activity[0]
    assert {key: activity[key] for key in activity if key != "duration_ms"} == {
        "tool_call_id": "call_1",
        "name": "search_documents",
        "arguments": {"query": "invoice"},
        "summary": "Found 1 matching document(s).",
        "preview": "Doc 42 matched.",
    }
    assert activity["duration_ms"] >= 0
    assert any(event["type"] == "tool_call_started" for event in events)
    assert any(event["type"] == "tool_call_completed" for event in events)
    assert any(
        event["type"] == "usage"
        and event["scope"] == "step"
        and event["model"] == "openai/chat-model"
        for event in events
    )
