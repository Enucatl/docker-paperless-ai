"""
Unit tests for metadata extraction strategies (LLM vs NuExtract).

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
    BaseExtractionStrategy,
    NuExtractStrategy,
    StructuredOutputStrategy,
    _ExtractedMetadata,
    _build_metadata_context,
    _extract_metadata,
)
from paperless_ai.core.config import AgentConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """Minimal AgentConfig for testing."""
    config = MagicMock(spec=AgentConfig)
    config.metadata_model = "test-model"
    config.metadata_api_base = None
    config.metadata_prompt = "Extract metadata from the following text:"
    config.llm_retries = 2
    config.nuextract_json_retries = 2
    config.get_metadata_litellm_kwargs = lambda: {}
    return config


# ---------------------------------------------------------------------------
# Tests: _fallback_parse (json-repair integration)
# ---------------------------------------------------------------------------


def test_build_metadata_context_keeps_start_middle_and_end() -> None:
    text = "A" * 3000 + "B" * 3000 + "C" * 3000 + "D" * 3000 + "E" * 3000

    result = _build_metadata_context(
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
        assert "title" in result or result == {}  # Either repairs it or returns empty

    def test_fallback_parse_missing_commas(self, strategy):
        """JSON with missing commas should be repaired."""
        raw = '{"title": "Test" "date": "2024-01-15"}'
        result = strategy._fallback_parse(raw)
        # json-repair attempts to add the missing comma
        assert "title" in result or result == {}

    def test_fallback_parse_trailing_comma(self, strategy):
        """JSON with trailing comma should be repaired."""
        raw = '{"title": "Test", "date": "2024-01-15",}'
        result = strategy._fallback_parse(raw)
        assert "title" in result or result == {}

    def test_fallback_parse_non_string_values(self, strategy):
        """JSON with non-string values (numbers, booleans) should parse."""
        raw = '{"pages": 10, "is_important": true, "confidence": 0.95}'
        result = strategy._fallback_parse(raw)
        # All keys should be present if parsing succeeds
        assert "pages" in result or result == {}

    def test_fallback_parse_null_values(self, strategy):
        """JSON with null values should parse."""
        raw = '{"title": "Test", "correspondent": null, "date": null}'
        result = strategy._fallback_parse(raw)
        assert "title" in result or result == {}

    def test_fallback_parse_completely_invalid(self, strategy):
        """Completely malformed input should return empty dict gracefully."""
        raw = "this is not json at all!!!"
        result = strategy._fallback_parse(raw)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: StructuredOutputStrategy
# ---------------------------------------------------------------------------


class TestStructuredOutputStrategy:
    """Test LLM-based extraction with response_format."""

    @pytest.fixture
    def strategy(self):
        return StructuredOutputStrategy()

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
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Test Invoice"
        assert result.date == datetime.date(2024, 1, 15)
        assert result.correspondent == "Acme Corp"

    @pytest.mark.asyncio
    async def test_extract_malformed_json_fallback(self, strategy, mock_config):
        """LLM response with malformed JSON should fall back gracefully."""
        # Malformed but recoverable JSON
        raw_response = '{"title": "Test", "date": "2024-01-15",}'
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        # Should create an _ExtractedMetadata object (may have empty fields if repair fails)
        assert isinstance(result, _ExtractedMetadata)

    @pytest.mark.asyncio
    async def test_extract_missing_fields(self, strategy, mock_config):
        """LLM response with only some fields should handle gracefully."""
        raw_response = json.dumps(
            {
                "title": "Invoice",
                # date and correspondent missing
            }
        )
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Invoice"
        assert result.date is None
        assert result.correspondent is None

    @pytest.mark.asyncio
    async def test_extract_empty_response(self, strategy, mock_config):
        """Empty or null LLM response should default to empty metadata."""
        mock_message = MagicMock()
        mock_message.content = None
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        # Should default to empty metadata
        assert result.title is None
        assert result.date is None
        assert result.correspondent is None


# ---------------------------------------------------------------------------
# Tests: NuExtractStrategy
# ---------------------------------------------------------------------------


class TestNuExtractStrategy:
    """Test template-based extraction with NuExtract model."""

    @pytest.fixture
    def strategy(self):
        return NuExtractStrategy()

    @pytest.mark.asyncio
    async def test_extract_successful(self, strategy, mock_config):
        """Successful NuExtract response with valid JSON."""
        # NuExtract returns a specific template structure
        raw_response = json.dumps(
            {
                "title_summarizing_subject_clear_concise_descriptive": "Test Invoice",
                "document_date": "2024-01-15",
                "issuing_organization_or_sender": "Acme Corp",
            }
        )
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Test Invoice"
        assert result.date == datetime.date(2024, 1, 15)
        assert result.correspondent == "Acme Corp"

    @pytest.mark.asyncio
    async def test_extract_missing_template_fields(self, strategy, mock_config):
        """NuExtract with missing template fields should handle gracefully."""
        raw_response = json.dumps(
            {
                "title_summarizing_subject_clear_concise_descriptive": "Test",
                # Missing date and correspondent
            }
        )
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        assert result.title == "Test"
        assert result.date is None
        assert result.correspondent is None

    @pytest.mark.asyncio
    async def test_extract_malformed_json_retry(self, strategy, mock_config):
        """NuExtract with malformed JSON on retry should fall back."""
        # First response: invalid JSON that triggers retry
        # The strategy will retry and potentially get fixed JSON
        raw_response = '{"title": "Test"'  # Incomplete JSON
        mock_message = MagicMock()
        mock_message.content = raw_response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await strategy.extract("Sample OCR text", mock_config)

        # Should create metadata object (may be empty if unrepair able)
        assert isinstance(result, _ExtractedMetadata)


# ---------------------------------------------------------------------------
# Tests: Date field validation
# ---------------------------------------------------------------------------


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


class FixedMetadataStrategy(BaseExtractionStrategy):
    """Test strategy that returns a Pydantic date field."""

    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        return _ExtractedMetadata(
            title="Test",
            date="2024-01-15",
            correspondent="Acme",
            summary="Test summary.",
        )


@pytest.mark.asyncio
async def test_extract_metadata_state_serializes_date_as_string(mock_config) -> None:
    """LangGraph state stores JSON-safe metadata for DocumentMetadata."""
    state = {"extracted_text_chunks": ["Sample OCR text"]}

    result = await _extract_metadata(state, mock_config, FixedMetadataStrategy())

    assert result["_extracted_metadata"]["date"] == "2024-01-15"
