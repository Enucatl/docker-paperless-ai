"""Parse complete PDFs through layout-parsing, then extract document metadata."""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Literal, Optional

import datetime as _dt

from json_repair import repair_json
from pydantic import BaseModel, Field, field_validator

from paperless_ai.agents.base import AgentResult, BaseDocumentAgent, DocumentMetadata
from paperless_ai.agents.state import AgentState
from paperless_ai.core.config import AgentConfig
from paperless_ai.inference import complete
from paperless_common.languages import LANGUAGES

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


def build_metadata_document_context(
    text: str,
    *,
    max_chars: int | None = None,
    start_chars: int = 2500,
    end_chars: int = 2000,
    middle_windows: int = 3,
) -> str:
    """Build the context used by metadata extraction.

    By default, preserve the complete OCR transcript. A character limit can be
    supplied by callers that explicitly need bounded context; the evaluation
    path deliberately does not use one.
    """
    if max_chars is None or len(text) <= max_chars:
        return text

    middle_budget = max(0, max_chars - start_chars - end_chars)
    window_count = max(0, middle_windows if middle_budget else 0)
    window_size = middle_budget // window_count if window_count else 0

    ranges: list[tuple[int, int]] = [(0, min(start_chars, len(text)))]
    middle_start = ranges[0][1]
    middle_end = max(middle_start, len(text) - end_chars)
    middle_span = middle_end - middle_start
    if window_size > 0 and middle_span > 0:
        for idx in range(window_count):
            center = middle_start + round((idx + 1) * middle_span / (window_count + 1))
            start = max(middle_start, center - (window_size // 2))
            end = min(middle_end, start + window_size)
            start = max(middle_start, end - window_size)
            if start < end:
                ranges.append((start, end))
    ranges.append((max(0, len(text) - end_chars), len(text)))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))

    parts: list[str] = []
    last_end = 0
    for start, end in merged:
        if start > last_end:
            parts.append("\n...\n")
        parts.append(text[start:end])
        last_end = end
    return "".join(parts)


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


# ---------------------------------------------------------------------------
# Extraction Strategy Pattern: separate LLM extraction from orchestration
# ---------------------------------------------------------------------------


class BaseExtractionStrategy(ABC):
    """Abstract base for metadata extraction strategies."""

    def _fallback_parse(self, raw: str) -> dict:
        """Robust fallback parser for handling malformed JSON output.

        Uses json-repair library to salvage broken JSON from LLM outputs.
        Handles escaped quotes, non-string values, missing commas, trailing commas, etc.
        If json-repair fails, returns an empty dict rather than raising.
        """
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # Use json-repair to salvage malformed JSON
        try:
            repaired = repair_json(raw)
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            log.debug("Could not repair JSON output: %s", raw[:200])
            return {}

    @abstractmethod
    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        """Extract metadata from text using this strategy."""
        pass


class StructuredOutputStrategy(BaseExtractionStrategy):
    """Standard LLM extraction using JSON schema / response_format."""

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
            "num_retries": config.llm_retries,
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
        raw = _get_completion_text(response) or "{}"
        log.info("Smart agent: metadata raw response: %s", raw)

        # Try strict Pydantic validation first, then fall back to loose parsing
        try:
            return _ExtractedMetadata.model_validate_json(raw)
        except Exception:
            data = self._fallback_parse(raw)
            try:
                return _ExtractedMetadata.model_validate(data)
            except Exception:
                # If date is unparseable, drop it and keep the rest
                data.pop("date", None)
                return _ExtractedMetadata.model_validate(data)


def _select_extraction_strategy(config: AgentConfig) -> BaseExtractionStrategy:
    """Select the appropriate extraction strategy based on the configured metadata model."""
    if "nuextract" in config.metadata_model.lower():
        return NuExtractStrategy()
    return StructuredOutputStrategy()


class NuExtractStrategy(BaseExtractionStrategy):
    """NuExtract template-based extraction using extra_body."""

    # NuExtract template with highly descriptive keys
    document_title_key = "title_summarizing_subject_clear_concise_descriptive"
    date_key = "document_date"
    correspondent_key = "issuing_organization_or_sender"
    _NUEXTRACT_TEMPLATE = json.dumps(
        {
            document_title_key: "string",
            date_key: "date-time",
            correspondent_key: "string",
            "languages": ["string"],
        },
        indent=4,
    )

    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        """Use NuExtract template passed via extra_body.

        Retries up to ``config.nuextract_json_retries`` times when the model
        returns invalid JSON before falling back to heuristic parsing.
        Temperature increases from 0 to 0.1 on retries to encourage variation.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "languages: "
                            + _ExtractedMetadata.model_fields["languages"].description
                            + "\n\nDocument:\n"
                            + text
                        ),
                    }
                ],
            }
        ]
        template_str = json.dumps(json.loads(self._NUEXTRACT_TEMPLATE), indent=4)
        base_kwargs: dict = {
            "model": config.metadata_model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"template": template_str}},
            "num_retries": config.llm_retries,
        }

        from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

        max_attempts = max(1, config.nuextract_json_retries)
        raw = "{}"

        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                retry=retry_if_exception_type(json.JSONDecodeError),
                reraise=True,
            ):
                with attempt:
                    # Increase temperature on retries to encourage variation.
                    # Without this, temperature=0 produces identical output every
                    # attempt. 0.1 adds just enough stochasticity.
                    attempt_num = attempt.retry_state.attempt_number
                    kwargs = {
                        **base_kwargs,
                        "temperature": 0 if attempt_num == 1 else 0.1,
                    }
                    response = await complete(
                        model=kwargs.pop("model"),
                        messages=kwargs.pop("messages"),
                        endpoint=config.metadata_endpoint,
                        domain="metadata_extraction",
                        **kwargs,
                    )
                    raw = _get_completion_text(response) or "{}"
                    log.info(
                        "Smart agent: NuExtract raw (attempt %d/%d): %s",
                        attempt_num,
                        max_attempts,
                        raw,
                    )
                    data = json.loads(raw)
                    return _ExtractedMetadata.model_validate(
                        {
                            "title": data.get(self.document_title_key),
                            "date": data.get(self.date_key),
                            "correspondent": data.get(self.correspondent_key),
                            "languages": data.get("languages"),
                        }
                    )
        except json.JSONDecodeError:
            pass

        # All retries exhausted — fall back to heuristic parsing
        log.warning(
            "Smart agent: NuExtract returned invalid JSON after %d attempts, using heuristic fallback",
            max_attempts,
        )
        data = self._fallback_parse(raw)
        return _ExtractedMetadata.model_validate(
            {
                "title": data.get(self.document_title_key),
                "date": data.get(self.date_key),
                "correspondent": data.get(self.correspondent_key),
                "languages": data.get("languages"),
            }
        )


# ---------------------------------------------------------------------------
# Graph node implementations (plain async functions — no class needed)
# ---------------------------------------------------------------------------


async def _extract_metadata(
    state: AgentState, config: AgentConfig, strategy: BaseExtractionStrategy
) -> dict:
    """Node 4: Join all text chunks and extract metadata using the provided strategy."""
    chunks = state["extracted_text_chunks"]
    full_text = "\n\n".join(chunks)
    if not full_text.strip():
        raise ValueError("OCR returned an empty transcript")

    # Use the strategy to extract metadata
    metadata_context = build_metadata_document_context(full_text)
    extracted = await strategy.extract(metadata_context, config)

    # Store final metadata back into state for the agent to read after graph completion
    return {
        "_extracted_metadata": extracted.model_dump(mode="json"),
        "_full_text": full_text,
        "_metadata_context": metadata_context,
    }


class SmartDocumentAgent(BaseDocumentAgent):
    """Complete-document parsing followed by metadata extraction."""

    def __init__(
        self,
        config: AgentConfig,
        extraction_strategy: Optional[BaseExtractionStrategy] = None,
    ):
        self._config = config
        self._strategy = extraction_strategy or StructuredOutputStrategy()
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as e:
            raise ImportError(
                "langgraph is required for SmartDocumentAgent: pip install langgraph"
            ) from e

        config = self._config
        strategy = self._strategy

        async def parse_document(state: AgentState) -> dict:
            """Use the same complete-document parser as the production worker."""
            from paperless_ai.agents.paddle_ocr import run_paddle_ocr

            text, _, pages, _ = await run_paddle_ocr(state["file_path"], config)
            return {"extracted_text_chunks": [text], "total_pages": pages}

        async def extract_metadata(state: AgentState) -> dict:
            """Extract metadata from the complete transcript."""
            return await _extract_metadata(state, config, strategy)

        workflow = StateGraph(AgentState)
        workflow.add_node("parse_document", parse_document)
        workflow.add_node("extract_metadata", extract_metadata)
        workflow.set_entry_point("parse_document")
        workflow.add_edge("parse_document", "extract_metadata")
        workflow.add_edge("extract_metadata", END)

        return workflow.compile()

    async def process(self, file_path: str, existing_hints: dict) -> AgentResult:
        t_start = time.time()
        initial_state: AgentState = {
            "file_path": file_path,
            "total_pages": 0,
            "extracted_text_chunks": [],
        }
        final_state = await self._graph.ainvoke(initial_state)

        # Retrieve results stored by extract_metadata node
        extracted_dict = final_state.get("_extracted_metadata", {})
        full_text = final_state.get(
            "_full_text", "\n\n".join(final_state.get("extracted_text_chunks", []))
        )
        metadata_context = final_state.get(
            "_metadata_context", build_metadata_document_context(full_text)
        )

        log.info(
            "Smart agent: done — title=%r date=%r correspondent=%r",
            extracted_dict.get("title"),
            extracted_dict.get("date"),
            extracted_dict.get("correspondent"),
        )

        metadata = DocumentMetadata(
            title=extracted_dict.get("title"),
            document_date=extracted_dict.get("date"),
            correspondent=extracted_dict.get("correspondent"),
            summary=extracted_dict.get("summary"),
            languages=extracted_dict.get("languages"),
            full_ocr_transcript=full_text,
        )

        return AgentResult(
            metadata=metadata,
            metadata_context=metadata_context,
            elapsed_s=round(time.time() - t_start, 1),
            pages=final_state.get("total_pages", 0),
            chars=len(full_text),
            ocr_method="layout-parsing",
        )
