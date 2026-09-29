"""
Unit tests for metadata extraction.

Tests validate graceful handling of:
- Successful JSON extraction
- Malformed JSON (escaped quotes, missing commas, trailing commas, etc.)
- Missing fields in JSON
- Empty/null values
- Non-string values (numbers, booleans)
"""

import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from paperless_ai.agents.smart_graph_agent import (
    SmartDocumentAgent,
    StructuredOutputStrategy,
    _ExtractedMetadata,
    build_metadata_document_context,
)
from paperless_ai.core.config import AgentConfig
from shared_inference import CompletionResult, Usage


def completion_result(content: str | None) -> CompletionResult:
    return CompletionResult(
        content=content,
        message={"role": "assistant", "content": content},
        tool_calls=[],
        reasoning=None,
        usage=Usage(),
        request_id=None,
        raw={},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Minimal AgentConfig for testing."""
    config = MagicMock(spec=AgentConfig)
    config.metadata_model = "test-model"
    config.metadata_endpoint = None
    config.metadata_prompt = "Extract metadata from the following text:"
    config.metadata_response_format = "auto"
    config.llm_retries = 2
    config.get_metadata_kwargs = lambda: {}
    return config


# ---------------------------------------------------------------------------
# Tests: _fallback_parse (json-repair integration)
# ---------------------------------------------------------------------------


def test_build_metadata_context_keeps_start_middle_and_end() -> None:
    text = "A" * 3000 + "B" * 3000 + "C" * 3000 + "D" * 3000 + "E" * 3000

    result = build_metadata_document_context(
        text,
        max_chars=1000,
        start_chars=200,
        end_chars=200,
        middle_windows=3,
    )

    assert result.startswith("A" * 200)
    assert "B" * 100 in result
    assert "C" * 100 in result
    assert "D" * 100 in result
    assert result.endswith("E" * 200)
    assert len(result) <= 1020


def test_metadata_response_format_defaults_to_auto() -> None:
    config = AgentConfig(metadata_model="metadata", chat_model="chat")

    assert config.metadata_response_format == "auto"


def test_metadata_response_format_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("INFERENCE_METADATA_RESPONSE_FORMAT", "none")

    config = AgentConfig(metadata_model="metadata", chat_model="chat")

    assert config.metadata_response_format == "none"


def test_metadata_response_format_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        AgentConfig(
            metadata_model="metadata",
            chat_model="chat",
            metadata_response_format="invalid",
        )


class TestFallbackParsing:
    """Test the _fallback_parse method with malformed JSON."""

    @pytest.fixture
    def strategy(self):
        """Use StructuredOutputStrategy to test the base _fallback_parse."""
        return StructuredOutputStrategy()

    def test_fallback_parse_valid_json(self, strategy):
        """Valid JSON should parse successfully."""
        raw = '{"title": "Test Invoice", "date": "2024-01-15", "correspondent": "Acme"}'
        result = strategy._fallback_parse(raw)
        assert result == {
            "title": "Test Invoice",
            "date": "2024-01-15",
            "correspondent": "Acme",
        }

    def test_fallback_parse_escaped_quotes(self, strategy):
        """JSON with escaped quotes in values should be repaired."""
        raw = r'{"title": "Invoice from \"Acme Corp\"", "date": "2024-01-15"}'
        result = strategy._fallback_parse(raw)
        # json-repair should handle this gracefully
        assert result["title"] == 'Invoice from "Acme Corp"'

    def test_fallback_parse_missing_commas(self, strategy):
        """JSON with missing commas should be repaired."""
        raw = '{"title": "Test" "date": "2024-01-15"}'
        result = strategy._fallback_parse(raw)
        # json-repair attempts to add the missing comma
        assert result["title"] == "Test"

    def test_fallback_parse_trailing_comma(self, strategy):
        """JSON with trailing comma should be repaired."""
        raw = '{"title": "Test", "date": "2024-01-15",}'
        result = strategy._fallback_parse(raw)
        assert result["title"] == "Test"

    def test_fallback_parse_non_string_values(self, strategy):
        """JSON with non-string values (numbers, booleans) should parse."""
        raw = '{"pages": 10, "is_important": true, "confidence": 0.95}'
        result = strategy._fallback_parse(raw)
        # All keys should be present if parsing succeeds
        assert result == {"pages": 10, "is_important": True, "confidence": 0.95}

    def test_fallback_parse_null_values(self, strategy):
        """JSON with null values should parse."""
        raw = '{"title": "Test", "correspondent": null, "date": null}'
        result = strategy._fallback_parse(raw)
        assert result["title"] == "Test"

    def test_fallback_parse_completely_invalid(self, strategy):
        """Unrepairable output must fail extraction."""
        raw = "this is not json at all!!!"
        with pytest.raises(ValueError):
            strategy._fallback_parse(raw)


# ---------------------------------------------------------------------------
# Tests: StructuredOutputStrategy
# ---------------------------------------------------------------------------


class TestStructuredOutputStrategy:
    """Test LLM-based extraction with response_format."""

    @pytest.fixture
    def strategy(self):
        return StructuredOutputStrategy()

    @pytest.mark.parametrize(
        ("policy", "expected_response_format", "has_instructions"),
        [
            ("auto", "schema", True),
            ("json_schema", "schema", True),
            ("json_object", {"type": "json_object"}, True),
            ("none", None, True),
        ],
    )
    @pytest.mark.asyncio
    async def test_response_format_policy_controls_prompt_and_kwargs(
        self,
        strategy,
        mock_config,
        policy,
        expected_response_format,
        has_instructions,
    ):
        """Each response format policy selects its prompt and request format."""
        mock_config.metadata_response_format = policy
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete",
            new_callable=AsyncMock,
        ) as mock_llm:
            mock_llm.return_value = completion_result('{"title": "Test"}')
            await strategy.extract("Sample OCR text", mock_config)

        kwargs = mock_llm.await_args.kwargs
        system_prompt = kwargs["messages"][0]["content"]
        assert ("Output JSON with these fields:" in system_prompt) is has_instructions
        if expected_response_format is None:
            assert "response_format" not in kwargs
        elif expected_response_format == "schema":
            assert kwargs["response_format"]["type"] == "json_schema"
            json_schema = kwargs["response_format"]["json_schema"]
            assert json_schema["strict"] is True
            schema = json_schema["schema"]
            assert schema["required"] == list(schema["properties"])
            assert schema["additionalProperties"] is False
            assert all(
                "default" not in property_schema
                for property_schema in schema["properties"].values()
            )
        else:
            assert kwargs["response_format"] == expected_response_format

    @pytest.mark.asyncio
    async def test_extract_successful_json(self, strategy, mock_config):
        """Successful LLM response with valid JSON."""
        raw_response = json.dumps(
            {
                "title": "Test Invoice",
                "date": "2024-01-15",
                "correspondent": "Acme Corp",
            }
        )
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = completion_result(raw_response)
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Test Invoice"
        assert result.date == datetime.date(2024, 1, 15)
        assert result.correspondent == "Acme Corp"

    @pytest.mark.asyncio
    async def test_extract_malformed_json_fallback(self, strategy, mock_config):
        """LLM response with malformed JSON should fall back gracefully."""
        # Malformed but recoverable JSON
        raw_response = '{"title": "Test", "date": "2024-01-15",}'
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = completion_result(raw_response)
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Test"
        assert result.date == datetime.date(2024, 1, 15)

    @pytest.mark.asyncio
    async def test_extract_missing_fields(self, strategy, mock_config):
        """LLM response with only some fields should handle gracefully."""
        raw_response = json.dumps(
            {
                "title": "Invoice",
                # date and correspondent missing
            }
        )
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = completion_result(raw_response)
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Invoice"
        assert result.date is None
        assert result.correspondent is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            " ",
            "not JSON",
            "{}",
            "{,}",
            "[]",
            '[{"title": "Test"}]',
            "null",
            "42",
            '"title"',
            '{"unknown": "value"}',
            '{"title": 42}',
            '{"date": "not a date"}',
        ],
    )
    async def test_extract_invalid_response(self, strategy, mock_config, raw):
        """Invalid responses must reach the stage failure/retry boundary."""
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete",
            new=AsyncMock(return_value=completion_result(raw)),
        ):
            with pytest.raises(ValueError):
                await strategy.extract("Sample OCR text", mock_config)

    @pytest.mark.asyncio
    async def test_extract_recovers_invalid_optional_date(self, strategy, mock_config):
        """An invalid date does not discard otherwise valid metadata."""
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete",
            new=AsyncMock(
                return_value=completion_result(
                    '{"title": "Invoice", "date": "not a date"}'
                )
            ),
        ):
            result = await strategy.extract("Sample OCR text", mock_config)
        assert result.title == "Invoice"
        assert result.date is None

    @pytest.mark.asyncio
    async def test_extract_optional_null_fields(self, strategy, mock_config):
        """Explicitly unknown optional fields remain valid metadata."""
        with patch(
            "paperless_ai.agents.smart_graph_agent.complete",
            new=AsyncMock(
                return_value=completion_result(
                    '{"title": null, "date": null, "summary": null, "correspondent": null}'
                )
            ),
        ):
            result = await strategy.extract("Sample OCR text", mock_config)
        assert result.title is None
        assert result.date is None
        assert result.summary is None
        assert result.correspondent is None


# ---------------------------------------------------------------------------
# Tests: Date field validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"languages": [" EN ", "de", "DE", "gsw"]}, ["en"]),
        ({"languages": ["xx"]}, ["und"]),
        ({"languages": ["yy"]}, ["und"]),
        ({"languages": ["gsw"]}, ["und"]),
        ({"languages": ["CMN"]}, ["cmn"]),
        ({}, ["und"]),
        ({"languages": None}, ["und"]),
        ({"languages": []}, ["und"]),
        ({"languages": "en"}, ["en"]),
        ({"languages": {"en": True}}, ["und"]),
        (
            {"languages": [None, 42, True, {}, [], "", "English", "e", "e1", "éé"]},
            ["und"],
        ),
        ({"languages": ["en", "de-DE", " français ", False, "FR"]}, ["en"]),
    ],
)
@pytest.mark.asyncio
async def test_languages_survive_extraction_and_fallback(
    malformed, fields, expected, mock_config
) -> None:
    """Extraction keeps one primary code and uses und when unknown."""
    raw = json.dumps({"title": "Test", **fields})
    if malformed:
        raw = raw[:-1] + ",}"
    with patch(
        "paperless_ai.agents.smart_graph_agent.complete",
        new=AsyncMock(return_value=completion_result(raw)),
    ) as complete:
        result = await StructuredOutputStrategy().extract(
            "German and English OCR text", mock_config
        )

    assert result.languages == expected
    kwargs = complete.await_args.kwargs
    prompt = kwargs["messages"][0]["content"]
    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert "languages" in schema["required"]
    from paperless_common.languages import LANGUAGES

    assert set(schema["properties"]["languages"]["items"]["enum"]) == set(LANGUAGES)
    assert len(LANGUAGES) == 51
    assert "Exactly one code" in prompt
    assert "ISO 639-1" in prompt and "ISO 639-3" in prompt
    assert "incidental foreign names and isolated words" in prompt
    assert "use 'und' when the language is undefined or unclear" in prompt


class TestDateFieldValidation:
    """Test date field parsing and validation in _ExtractedMetadata."""

    def test_valid_iso_date(self):
        """Valid ISO date string is coerced to a date object by Pydantic."""
        meta = _ExtractedMetadata(
            title="Test",
            date="2024-01-15",
            correspondent="Acme",
        )
        assert meta.date == datetime.date(2024, 1, 15)

    def test_iso_datetime_with_time_rejected(self):
        """ISO datetime with non-zero time is rejected by Pydantic date validation."""
        with pytest.raises(ValidationError):
            _ExtractedMetadata(
                title="Test",
                date="2024-01-15T10:30:00",
                correspondent="Acme",
            )

    def test_invalid_date_raises(self):
        """Invalid date strings raise a ValidationError."""
        with pytest.raises(ValidationError):
            _ExtractedMetadata(
                title="Test",
                date="not a date",
                correspondent="Acme",
            )


class FixedMetadataStrategy:
    """Test strategy that returns a Pydantic date field."""

    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        return _ExtractedMetadata(
            title="Test",
            date="2024-01-15",
            correspondent="Acme",
            summary="Test summary.",
            languages=[" EN ", "de"],
        )


@pytest.mark.asyncio
async def test_metadata_propagates_to_public_agent_result(mock_config) -> None:
    """Direct extraction preserves fields, serializes dates, and reports OCR stats."""
    with patch(
        "paperless_ai.agents.paddle_ocr.run_paddle_ocr",
        new=AsyncMock(return_value=("Stored OCR text", {}, 3, 0.1)),
    ):
        result = await SmartDocumentAgent(mock_config, FixedMetadataStrategy()).process(
            "unused.pdf", {}
        )

    assert result.metadata.title == "Test"
    assert result.metadata.document_date == "2024-01-15"
    assert result.metadata.correspondent == "Acme"
    assert result.metadata.summary == "Test summary."
    assert result.metadata.languages == ["en"]
    assert result.metadata.full_ocr_transcript == "Stored OCR text"
    assert result.metadata_context == "Stored OCR text"
    assert result.pages == 3
    assert result.chars == len("Stored OCR text")


@pytest.mark.asyncio
async def test_metadata_context_is_the_complete_strategy_input(mock_config) -> None:
    """Jev and extraction receive the same complete OCR transcript."""
    text = "A" * 7000
    strategy = StructuredOutputStrategy()
    with (
        patch(
            "paperless_ai.agents.paddle_ocr.run_paddle_ocr",
            new=AsyncMock(return_value=(text, {}, 1, 0.1)),
        ),
        patch.object(
            strategy, "extract", new=AsyncMock(return_value=_ExtractedMetadata())
        ) as extract,
    ):
        result = await SmartDocumentAgent(mock_config, strategy).process(
            "unused.pdf", {}
        )

    extract.assert_awaited_once_with(text, mock_config)
    assert result.metadata_context == text
    assert result.metadata.document_date is None


@pytest.mark.asyncio
async def test_empty_ocr_fails_before_extraction(mock_config) -> None:
    """Do not send an empty OCR transcript to metadata inference."""
    strategy = StructuredOutputStrategy()
    with (
        patch(
            "paperless_ai.agents.paddle_ocr.run_paddle_ocr",
            new=AsyncMock(return_value=(" \n", {}, 1, 0.1)),
        ),
        patch.object(strategy, "extract", new=AsyncMock()) as extract,
    ):
        with pytest.raises(ValueError, match="OCR returned an empty transcript"):
            await SmartDocumentAgent(mock_config, strategy).process("unused.pdf", {})

    extract.assert_not_awaited()


def test_metadata_reasoning_default_is_independent_of_ocr(monkeypatch) -> None:
    """Keep minimal metadata reasoning after removing legacy OCR settings."""
    for key in (
        "INFERENCE_METADATA_REASONING_EFFORT",
        "METADATA_REASONING_EFFORT",
        "metadata_reasoning_effort",
        "INFERENCE_METADATA_EXTRA_KWARGS",
        "METADATA_EXTRA_KWARGS",
        "metadata_extra_kwargs",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("INFERENCE_OCR_REASONING_EFFORT", "high")
    config = AgentConfig(metadata_model="metadata", chat_model="chat")
    assert config.get_metadata_kwargs()["reasoning_effort"] == "minimal"
    config.metadata_reasoning_effort = "high"
    assert config.get_metadata_kwargs()["reasoning_effort"] == "high"
    config.metadata_reasoning_effort = None
    assert "reasoning_effort" not in config.get_metadata_kwargs()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "{}", "not JSON", '[{"title": "Test"}]'])
async def test_invalid_metadata_retries_without_writing(monkeypatch, mock_config, raw):
    """An unusable model response cannot clear fields or complete the stage."""
    from paperless_ai.core import runner
    from paperless_common.queue import TaskQueues

    runner.clear_shutdown_request()
    mock_config.ocr_concurrency = 1
    mock_config.stage_max_attempts = 3
    mock_config.tag_metadata = "ai:run-metadata"
    client = AsyncMock()
    client.get_document_with_content.return_value = {
        "id": 42,
        "content": "Invoice OCR",
        "tags": [7],
        "custom_fields": [{"field": 2, "value": "Existing summary"}],
    }
    queues = AsyncMock()
    queues.stage_size.return_value = 1
    queues.peek_stage.return_value = {42}
    queues.mark_failure.return_value = (1, False)
    monkeypatch.setattr(
        "paperless_ai.agents.smart_graph_agent.complete",
        AsyncMock(return_value=completion_result(raw)),
    )

    assert await runner.run_metadata_batch(client, mock_config, queues, 1, 2, 3) == (
        0,
        1,
    )
    client.patch_document.assert_not_awaited()
    queues.remove.assert_not_awaited()
    queues.mark_failure.assert_awaited_once_with(
        42, TaskQueues.KEY_METADATA, max_attempts=3
    )
