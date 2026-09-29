from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import niquests

from paperless_common.telemetry import start_span
from paperless_ai.search.chat_agent import ChatCopilot, _extract_usage
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

    assert "Invalid tool arguments" in result.content
    assert "precision" in result.content and "recall" in result.content


def test_start_span_preserves_original_exception():
    with pytest.raises(ValueError, match="boom"):
        with start_span("paperless_ai.test.span"):
            raise ValueError("boom")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "languages",
    [
        {"it": 1361, "de": 700, "en": 305, "fr": 10, "th": 6, "pt": 3, "es": 1},
        {},
        niquests.ConnectionError("offline"),
    ],
)
async def test_chat_copilot_supplies_fresh_languages_before_model_call(languages):
    """Every turn gets current language context without persisting it in history."""
    config = MagicMock()
    config.get_chat_kwargs.return_value = {}
    client = AsyncMock()
    client.get_document_languages.side_effect = [languages, {"fr": 2}]
    client.metadata_snapshot = MagicMock(return_value=client)
    copilot = ChatCopilot(config=config, client=client)

    with patch(
        "paperless_ai.search.chat_agent.complete",
        new=AsyncMock(return_value=_completion("Answer")),
    ) as complete:
        first = await copilot.run_turn("Find invoices")
        first_prompt = complete.await_args.kwargs["messages"][0]["content"]
        expected = (
            ", ".join(f"{code}: {count} documents" for code, count in languages.items())
            if isinstance(languages, dict) and languages
            else "unknown"
        )
        assert (
            f"Observed document languages (ISO codes and document counts): {expected}"
            in first_prompt
        )
        assert "at most two languages per query" in first_prompt
        assert "even when there are matches" in first_prompt
        assert "remaining relevant observed languages" in first_prompt
        assert "only filter by language tags when the user requests it" in first_prompt
        assert all(message["role"] != "system" for message in first.history)

        await copilot.run_turn("Find more", history=first.history)
        second_prompt = complete.await_args.kwargs["messages"][0]["content"]
        assert second_prompt.endswith(
            "Observed document languages (ISO codes and document counts): fr: 2 documents."
        )
        assert client.get_document_languages.await_count == 2


@pytest.mark.asyncio
async def test_chat_copilot_run_turn_emits_events_and_aggregates_usage():
    config = MagicMock()
    config.chat_model = "openai/chat-model"
    config.metadata_model = "openai/metadata-model"
    config.chat_endpoint = None
    config.metadata_endpoint = None
    config.get_chat_kwargs.return_value = {}

    client = AsyncMock()
    client.get_document_languages.return_value = {"de": 700, "en": 305}
    client.metadata_snapshot = MagicMock(return_value=client)
    copilot = ChatCopilot(config=config, client=client)

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


@pytest.mark.parametrize("raw", ["[]", "42", '"text"', "null", "true", [], 42])
def test_parse_tool_arguments_rejects_non_objects(raw):
    with pytest.raises(ValueError, match="JSON object"):
        parse_tool_arguments(raw)


@pytest.mark.parametrize(
    "name, arguments",
    [
        ("read_full_document", {"doc_id": 1, "max_chars": value})
        for value in (499, 20001, "8000", True, None)
    ]
    + [
        ("search_documents", {"query": "invoice", "limit": value})
        for value in (0, 101, "20", True, 1.5)
    ]
    + [
        ("search_documents", {"query": 42}),
        ("search_documents", {"query": "invoice", "tags": [42]}),
        ("search_documents", {"query": "invoice", "year": "26"}),
        ("read_full_document", {}),
        ("read_full_document", {"doc_id": "1"}),
        ("get_available_metadata", {"extra": 1}),
        ("read_full_document", []),
        ("read_full_document", 42),
        ("unknown_tool", {}),
    ],
)
async def test_dispatch_rejects_invalid_arguments_without_client_calls(name, arguments):
    client = AsyncMock()
    result = await execute_tool_call_detailed(name, arguments, client=client)
    assert "Invalid tool arguments" in result.content
    assert not result.source_refs
    assert client.mock_calls == []


@pytest.mark.parametrize("max_chars", [500, 20000])
async def test_read_accepts_advertised_bounds(max_chars):
    client = AsyncMock()
    client.get_document_with_content.return_value = {
        "id": 1,
        "title": "Invoice",
        "content": "x" * 21000,
    }
    result = await execute_tool_call_detailed(
        "read_full_document", {"doc_id": 1, "max_chars": max_chars}, client=client
    )
    assert result.content == "[Doc 1 | Invoice]\n" + "x" * max_chars + "\n\n[truncated]"
    client.get_document_with_content.assert_awaited_once_with(1)


@pytest.mark.parametrize("limit", [1, 100])
async def test_search_accepts_advertised_bounds(limit):
    client = AsyncMock()
    client.search_documents_all.return_value = []
    await execute_tool_call_detailed(
        "search_documents",
        {"query": "invoice", "limit": limit, "mode": "recall"},
        client=client,
    )
    assert client.search_documents_all.await_args.kwargs["limit"] == limit


@pytest.mark.parametrize(
    "usage, expected",
    [
        (Usage(), None),
        (
            Usage(prompt_tokens=3),
            {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
        ),
        (
            Usage(completion_tokens=2),
            {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2},
        ),
        (
            Usage(total_tokens=7),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 7},
        ),
        (
            Usage(prompt_tokens=3, completion_tokens=2),
            {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        ),
        (
            Usage(total_tokens=0),
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        ),
    ],
)
def test_extract_usage_preserves_optional_counts(usage, expected):
    response = _completion("Answer")
    response.usage = usage
    assert _extract_usage(response) == expected


@pytest.mark.parametrize("arguments", ["{", "[]", "42", '{"doc_id":1,"max_chars":1}'])
async def test_chat_returns_validation_errors_to_model_then_recovers(arguments):
    config = MagicMock()
    config.get_chat_kwargs.return_value = {}
    client = AsyncMock()
    client.metadata_snapshot = MagicMock(return_value=client)
    client.get_document_languages.return_value = {}
    client.get_document_with_content.return_value = {"id": 1, "content": "Invoice"}

    def call(args, call_id):
        return _completion(
            "",
            [
                {
                    "id": call_id,
                    "function": {
                        "name": "read_full_document",
                        "arguments": args,
                    },
                }
            ],
        )

    events = []

    async def capture(event):
        events.append(event)

    with patch(
        "paperless_ai.search.chat_agent.complete",
        side_effect=[
            call(arguments, "invalid"),
            call('{"doc_id":1}', "valid"),
            _completion("Answer"),
        ],
    ):
        result = await ChatCopilot(config, client).run_turn(
            "Find invoice", event_callback=capture
        )
    results = [message for message in result.history if message["role"] == "tool"]
    assert "Invalid tool arguments" in results[0]["content"]
    assert "Invoice" in results[1]["content"]
    assert result.reply == "Answer"
    assert result.usage is None
    assert any(event["type"] == "usage" and not event["available"] for event in events)
    client.get_document_with_content.assert_awaited_once_with(1)
