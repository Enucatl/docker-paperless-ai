"""Parse complete PDFs through layout-parsing, then extract document metadata."""

import json
import logging
import re
import time
from typing import Literal, Optional

import datetime as _dt

from json_repair import repair_json
from pydantic import BaseModel, Field, ValidationError, field_validator

from paperless_ai.agents.base import AgentResult, DocumentMetadata
from paperless_ai.core.config import AgentConfig
from paperless_ai.inference import complete
from paperless_common.languages import LANGUAGES
from paperless_common.telemetry import start_span

log = logging.getLogger(__name__)


# Regex patterns for thinking tags emitted by some reasoning models
# (DeepSeek-R1, Qwen-QwQ, etc.) when thinking is not separated into
# reasoning_content by the provider/OpenAI-compatible inference.
_THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.DOTALL | re.IGNORECASE)


def _get_completion_text(response) -> str:
    """Return only the final answer text from a OpenAI-compatible inference completion response.

    OpenAI-compatible inference normalises reasoning providers so that:
      - ``message.reasoning_content`` holds the thinking trace (string).
      - ``message.content`` holds only the final answer.

    Two edge cases still need handling:

    1. **Block-list content** – Anthropic passes thinking as typed content
       blocks (``{"type": "thinking", ...}``).  When OpenAI-compatible inference surfaces this as
       a list we filter to ``type == "text"`` blocks only.

    2. **Inline tags** – Older OpenAI-compatible inference versions or models not yet normalised
       (DeepSeek-R1, Qwen-QwQ via Ollama, …) may prepend the thinking wrapped
       in ``<think>…</think>`` inside the content string.  We strip those.
    """
    message = response.message
    content = message.get("content")

    # Case 1: content is a list of typed blocks (Anthropic extended thinking).
    # Keep only text blocks; discard thinking blocks.
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        )

    content = content or ""

    # If OpenAI-compatible inference already separated thinking into reasoning_content the content
    # string is clean — return it directly.
    if response.reasoning:
        return content

    # Case 2: strip inline thinking tags for providers not yet normalised.
    content = _THINK_RE.sub("", content)
    return content.strip()


class _ExtractedMetadata(BaseModel):
    title: Optional[str] = Field(
        default=None,
        description=(
            "A clear, concise, and descriptive title summarizing the document's core subject "
            "or purpose for future retrieval. Maximum 100 characters. Use Title Case. "
            "Do NOT include full sentences, conversational text, disclaimers, or notes about the extraction process."
        ),
    )
    date: Optional[_dt.date] = Field(
        default=None, description="Primary document date as YYYY-MM-DD."
    )
    correspondent: Optional[str] = Field(
        default=None,
        description=(
            "Name of the company, institution, organization or person who sent, authored, or issued the document — not the recipient."
            "Prefer the name of the institution if the document is signed by a specific person on its behalf."
        ),
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "One or two sentences summarising the content and purpose for document search. "
            "Start directly with the subject matter, not with phrases like 'This document is', 'This is', "
            "or other meta framing. Be specific, factual, and useful for matching documents."
        ),
    )
    languages: list[Literal[*LANGUAGES]] = Field(
        default_factory=lambda: ["und"],
        min_length=1,
        max_length=1,
        description=(
            "Exactly one code for the document's primary substantive language, as a "
            "listed lowercase ISO 639-1 or ISO 639-3 code, exactly as given below. Ignore "
            "incidental foreign names and isolated words. Return a one-item list; "
            "use 'und' when the language is undefined or unclear or outside this list. "
            "Choose only from: "
            + "; ".join(f"{code}: {name}" for code, name in LANGUAGES.items())
        ),
    )

    @field_validator("languages", mode="before")
    @classmethod
    def normalize_languages(cls, value: object) -> list[str]:
        """Keep only the primary language, using the ISO undefined code as fallback."""
        candidates = value if isinstance(value, list) else [value]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip().lower() in LANGUAGES:
                return [candidate.strip().lower()]
        return ["und"]


def _field_instructions_from_schema() -> str:
    """Generate prompt instructions from _ExtractedMetadata field descriptions.

    Per Google's vLLM best practice: response_format enforces structure but the
    model never sees the schema descriptions — those must be in the system prompt.
    Always append this to the metadata system prompt regardless of which
    response_format tier is used.
    """
    props = _ExtractedMetadata.model_json_schema().get("properties", {})
    lines = ["Output JSON with these fields:"]
    for field_name, field_info in props.items():
        description = field_info.get("description", "")
        lines.append(f"- {field_name}: {description}")
    return "\n".join(lines)


def _metadata_response_format_tier(config: AgentConfig) -> tuple[str, object | None]:
    """Return the metadata prompt and the selected response format policy."""
    response_format_policy = config.metadata_response_format
    system_prompt = config.metadata_prompt + "\n\n" + _field_instructions_from_schema()
    metadata_schema = _ExtractedMetadata.model_json_schema()
    # OpenAI-compatible strict JSON Schema requires every property to be
    # required. Nullable fields preserve the distinction between "unknown" and
    # an omitted field in the model output.
    metadata_schema["required"] = list(metadata_schema.get("properties", {}))
    metadata_schema["additionalProperties"] = False
    for property_schema in metadata_schema.get("properties", {}).values():
        if isinstance(property_schema, dict):
            property_schema.pop("default", None)
    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "extracted-metadata",
            "strict": True,
            "schema": metadata_schema,
        },
    }
    if response_format_policy in {"json_schema", "auto"}:
        return system_prompt, schema_format

    if response_format_policy == "json_object":
        return system_prompt, {"type": "json_object"}
    if response_format_policy == "none":
        return system_prompt, None

    return system_prompt, None


class StructuredOutputStrategy:
    """Standard LLM extraction using JSON schema / response_format."""

    def _fallback_parse(self, raw: str) -> dict:
        """Parse a JSON object, repairing malformed JSON syntax when possible."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = json.loads(repair_json(raw))
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("Metadata response must be a non-empty JSON object")
        return parsed

    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        """Use standard LLM with response_format tiering."""
        system_prompt, response_format = _metadata_response_format_tier(config)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        kwargs: dict = {
            "model": config.metadata_model,
            "messages": messages,
            **config.get_metadata_kwargs(),
        }
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = await complete(
            model=kwargs.pop("model"),
            messages=kwargs.pop("messages"),
            endpoint=config.metadata_endpoint,
            domain="metadata_extraction",
            **kwargs,
        )
        raw = _get_completion_text(response)
        log.info("Smart agent: metadata raw response: %s", raw)
        data = self._fallback_parse(raw)
        if not _ExtractedMetadata.model_fields.keys() & data.keys():
            raise ValueError("Metadata response contains no metadata fields")
        try:
            return _ExtractedMetadata.model_validate(data)
        except ValidationError as exc:
            # Preserve the usable fields when only the optional date is invalid.
            if any(error["loc"] != ("date",) for error in exc.errors()):
                raise
            data.pop("date", None)
            if not _ExtractedMetadata.model_fields.keys() & data.keys():
                raise
            return _ExtractedMetadata.model_validate(data)


class SmartDocumentAgent:
    """Complete-document parsing followed by metadata extraction."""

    def __init__(
        self,
        config: AgentConfig,
        extraction_strategy: Optional[StructuredOutputStrategy] = None,
    ):
        self._config = config
        self._strategy = extraction_strategy or StructuredOutputStrategy()

    async def process(self, file_path: str) -> AgentResult:
        """Parse a document and return metadata with its full OCR context."""
        from paperless_ai.agents.paddle_ocr import run_paddle_ocr

        t_start = time.time()
        with start_span("parse_document"):
            full_text, _, pages, _ = await run_paddle_ocr(file_path, self._config)
        if not full_text.strip():
            raise ValueError("OCR returned an empty transcript")

        with start_span("extract_metadata"):
            extracted = await self._strategy.extract(full_text, self._config)

        log.info(
            "Smart agent: done — title=%r date=%r correspondent=%r",
            extracted.title,
            extracted.date,
            extracted.correspondent,
        )

        metadata = DocumentMetadata(
            title=extracted.title,
            document_date=extracted.date.isoformat() if extracted.date else None,
            correspondent=extracted.correspondent,
            summary=extracted.summary,
            languages=extracted.languages,
            full_ocr_transcript=full_text,
        )

        return AgentResult(
            metadata=metadata,
            metadata_context=full_text,
            elapsed_s=round(time.time() - t_start, 1),
            pages=pages,
            chars=len(full_text),
            ocr_method="layout-parsing",
        )
