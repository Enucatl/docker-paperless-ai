"""
SmartDocumentAgent: memory-safe LangGraph-based document processor.

Graph topology:

    analyze_pdf → batched_vision_ocr (loop until all pages done) → extract_metadata → END

Key design decisions:
- Vision OCR is always used (no native text extraction) for consistent quality
  across all PDF types. PyMuPDF native text is unreliable for scanned/mixed PDFs.
- PyMuPDF pixmaps are del'd immediately after base64 encoding to avoid leaks.
- LangGraph annotated state uses operator.add to accumulate text chunks without
  holding all images in memory simultaneously.
"""

import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import datetime as _dt

import fitz  # PyMuPDF
from json_repair import repair_json
from pydantic import BaseModel, Field

from paperless_ai.agents.base import AgentResult, BaseDocumentAgent, DocumentMetadata
from paperless_ai.agents.state import AgentState
from paperless_ai.core.config import AgentConfig
from paperless_ai.inference import complete

log = logging.getLogger(__name__)


class VisionOcrCache:
    """In-memory cache for deterministic evaluation OCR requests."""

    def __init__(self) -> None:
        """Initialize an empty OCR cache."""
        self._texts: dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def key(
        self,
        file_path: str,
        page_index: int,
        config: AgentConfig,
        language: str | None,
    ) -> str:
        """Build a cache key from the PDF and all OCR request parameters.

        Args:
            file_path: PDF path used for the OCR request.
            page_index: Zero-based PDF page index.
            config: Agent configuration supplying OCR parameters.
            language: Optional document language hint.

        Returns:
            A stable key that changes when the file or OCR request changes.
        """
        stat = Path(file_path).stat()
        return json.dumps(
            {
                "file": str(Path(file_path).resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "page": page_index,
                "model": config.ocr_model,
                "endpoint": config.ocr_endpoint,
                "prompt": config.ocr_prompt,
                "language": language,
                "max_image_dimension": config.ocr_max_image_dimension,
                "kwargs": config.get_ocr_kwargs(),
            },
            sort_keys=True,
            default=str,
        )

    def get(self, key: str) -> str | None:
        """Return cached OCR text for a key, if available."""
        text = self._texts.get(key)
        if text is None:
            self.misses += 1
        else:
            self.hits += 1
        return text

    def put(self, key: str, text: str) -> None:
        """Store OCR text under a request key."""
        self._texts[key] = text


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
                "content": [{"type": "text", "text": text}],
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
            }
        )


# ---------------------------------------------------------------------------
# Graph node implementations (plain async functions — no class needed)
# ---------------------------------------------------------------------------


def _select_ocr_pages(total_pages: int, config: "AgentConfig") -> list[int]:
    """Return the ordered list of 0-based page indices to send through vision OCR.

    For documents with total_pages <= ocr_page_limit_threshold every page is
    selected (no change in behaviour).  For longer documents only the first
    ocr_first_pages and last ocr_last_pages are selected — overlapping indices
    are deduplicated while preserving order.

    Rationale: Paperless-ngx Tesseract already produces full-document text for
    keyword search.  Vision OCR is only needed to capture semantically rich
    pages (cover, header, executive summary, signature block) for metadata
    extraction.
    """
    if total_pages <= config.ocr_page_limit_threshold:
        return list(range(total_pages))

    first = set(range(min(config.ocr_first_pages, total_pages)))
    last_start = max(total_pages - config.ocr_last_pages, 0)
    last = set(range(last_start, total_pages))
    return sorted(first | last)


def _count_pdf_pages(file_path: str) -> int:
    with fitz.open(file_path) as doc:
        return len(doc)


def _render_page_to_base64(file_path: str, page_idx: int, max_dim: int | None) -> str:
    with fitz.open(file_path) as doc:
        page = doc[page_idx]
        dpi = 300
        if max_dim:
            w_px = page.rect.width * dpi / 72
            h_px = page.rect.height * dpi / 72
            if max(w_px, h_px) > max_dim:
                dpi = int(dpi * max_dim / max(w_px, h_px))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        png_bytes = pix.tobytes("png")
        return base64.b64encode(png_bytes).decode()


async def _analyze_pdf(state: AgentState, config: AgentConfig) -> dict:
    """Node 1: Measure page count and select pages for vision OCR.

    Always uses vision OCR for consistent extraction quality across all PDFs.
    PyMuPDF native text extraction is unreliable for scanned documents and
    many digital PDFs where the embedded text is just metadata.
    """
    file_path = state["file_path"]
    total_pages = await asyncio.to_thread(_count_pdf_pages, file_path)

    ocr_page_indices = _select_ocr_pages(total_pages, config)

    if len(ocr_page_indices) < total_pages:
        log.info(
            "Smart agent: %d-page document — vision OCR limited to %d pages "
            "(first %d + last %d); threshold=%d",
            total_pages,
            len(ocr_page_indices),
            config.ocr_first_pages,
            config.ocr_last_pages,
            config.ocr_page_limit_threshold,
        )
    else:
        log.info("Smart agent: %d pages → vision OCR (all pages)", total_pages)

    return {
        "total_pages": total_pages,
        "ocr_page_indices": ocr_page_indices,
        "is_digital_text": False,  # Always use vision OCR
        "current_page": 0,
        "extracted_text_chunks": [],
    }


async def _batched_vision_ocr(
    state: AgentState,
    config: AgentConfig,
    ocr_cache: VisionOcrCache | None = None,
) -> dict:
    """Node 3: Render a batch of selected pages to base64 images and run vision OCR.

    current_page is an index into ocr_page_indices, not a raw page number.
    Pages are processed in configurable batches (vision_batch_size) so only a
    few images are held in memory at once. Pixmaps are freed immediately after
    encoding to avoid C-heap accumulation.
    """
    file_path = state["file_path"]
    current_idx = state["current_page"]
    batch_size = state["batch_size"]
    total_pages = state["total_pages"]
    page_indices = state["ocr_page_indices"]
    language = state.get("language")

    end_idx = min(current_idx + batch_size, len(page_indices))
    batch = page_indices[current_idx:end_idx]

    log.info(
        "Smart agent: vision OCR pages %s (of %d total)",
        ", ".join(str(p + 1) for p in batch),
        total_pages,
    )

    async def ocr_page(page_idx: int) -> str:
        cache_key = (
            ocr_cache.key(file_path, page_idx, config, language)
            if ocr_cache is not None
            else None
        )
        if cache_key is not None:
            cached_text = ocr_cache.get(cache_key)
            if cached_text is not None:
                log.info("Smart agent: OCR cache hit for page %d", page_idx + 1)
                return cached_text

        b64 = await asyncio.to_thread(
            _render_page_to_base64,
            file_path,
            page_idx,
            config.ocr_max_image_dimension,
        )

        prompt = config.ocr_prompt
        if language:
            prompt = f"The document language is primarily '{language}'. " + prompt

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        kwargs: dict = {
            "model": config.ocr_model,
            "messages": messages,
            "num_retries": config.llm_retries,
            **config.get_ocr_kwargs(),
        }
        response = await complete(
            model=kwargs.pop("model"),
            messages=kwargs.pop("messages"),
            endpoint=config.ocr_endpoint,
            domain="vision_ocr",
            **kwargs,
        )
        text = _get_completion_text(response)
        if cache_key is not None:
            ocr_cache.put(cache_key, text)
        return text

    chunks = await asyncio.gather(*(ocr_page(page_idx) for page_idx in batch))

    return {
        "extracted_text_chunks": chunks,
        "current_page": end_idx,  # advances as index into ocr_page_indices
    }


async def _extract_metadata(
    state: AgentState, config: AgentConfig, strategy: BaseExtractionStrategy
) -> dict:
    """Node 4: Join all text chunks and extract metadata using the provided strategy."""
    chunks = state["extracted_text_chunks"]
    full_text = "\n\n".join(chunks)

    # Use the strategy to extract metadata
    metadata_context = build_metadata_document_context(full_text)
    extracted = await strategy.extract(metadata_context, config)

    # Store final metadata back into state for the agent to read after graph completion
    return {
        "_extracted_metadata": extracted.model_dump(mode="json"),
        "_full_text": full_text,
        "_metadata_context": metadata_context,
    }


# ---------------------------------------------------------------------------
# OCR-only helper (no metadata extraction) for the decoupled pipeline
# ---------------------------------------------------------------------------


async def run_vision_ocr_only(
    file_path: str, config: "AgentConfig"
) -> tuple[str, int, float]:
    """Run only the vision OCR stage — no metadata extraction.

    Used by the OCR worker in the decoupled tag-driven pipeline.
    Bypasses LangGraph to avoid building a full graph for a single stage.

    Returns:
        (full_text, page_count, elapsed_seconds)
    """
    t_start = time.time()

    initial: AgentState = {
        "file_path": file_path,
        "language": None,
        "total_pages": 0,
        "ocr_page_indices": [],  # populated by _analyze_pdf
        "is_digital_text": False,
        "current_page": 0,
        "batch_size": config.vision_batch_size,
        "extracted_text_chunks": [],
    }

    analysis = await _analyze_pdf(initial, config)
    total_pages = analysis["total_pages"]
    state = {**initial, **analysis}
    all_chunks: list[str] = []

    while state["current_page"] < len(state["ocr_page_indices"]):
        update = await _batched_vision_ocr(state, config)
        all_chunks.extend(update["extracted_text_chunks"])
        state = {**state, "current_page": update["current_page"]}

    full_text = "\n\n".join(all_chunks)
    return full_text, total_pages, round(time.time() - t_start, 1)


# ---------------------------------------------------------------------------
# SmartDocumentAgent: wires nodes into a LangGraph
# ---------------------------------------------------------------------------


class SmartDocumentAgent(BaseDocumentAgent):
    """
    Memory-safe agentic document processor built on LangGraph.

    Routes to native text extraction for digital PDFs, or batched vision OCR
    for scanned/image PDFs. The vision path loops page-by-page in configurable
    batches so only `batch_size` pages are held in memory at any time.
    """

    def __init__(
        self,
        config: AgentConfig,
        extraction_strategy: Optional[BaseExtractionStrategy] = None,
        ocr_cache: VisionOcrCache | None = None,
    ):
        self._config = config
        self._strategy = extraction_strategy or StructuredOutputStrategy()
        self._ocr_cache = ocr_cache
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

        # Bind config and strategy into each node via closures
        async def analyze_pdf(state: AgentState) -> dict:
            return await _analyze_pdf(state, config)

        async def batched_vision_ocr(state: AgentState) -> dict:
            return await _batched_vision_ocr(state, config, self._ocr_cache)

        async def extract_metadata(state: AgentState) -> dict:
            return await _extract_metadata(state, config, strategy)

        def route_after_vision_ocr(state: AgentState) -> str:
            return (
                "batched_vision_ocr"
                if state["current_page"] < len(state["ocr_page_indices"])
                else "extract_metadata"
            )

        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_pdf", analyze_pdf)
        workflow.add_node("batched_vision_ocr", batched_vision_ocr)
        workflow.add_node("extract_metadata", extract_metadata)

        workflow.set_entry_point("analyze_pdf")
        workflow.add_edge("analyze_pdf", "batched_vision_ocr")
        workflow.add_conditional_edges(
            "batched_vision_ocr",
            route_after_vision_ocr,
            {
                "batched_vision_ocr": "batched_vision_ocr",
                "extract_metadata": "extract_metadata",
            },
        )
        workflow.add_edge("extract_metadata", END)

        return workflow.compile()

    async def process(self, file_path: str, existing_hints: dict) -> AgentResult:
        t_start = time.time()
        config = self._config

        initial_state: AgentState = {
            "file_path": file_path,
            "language": existing_hints.get("language"),
            "total_pages": 0,
            "ocr_page_indices": [],  # populated by _analyze_pdf node
            "is_digital_text": False,
            "current_page": 0,
            "batch_size": config.vision_batch_size,
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
            full_ocr_transcript=full_text,
        )

        return AgentResult(
            metadata=metadata,
            metadata_context=metadata_context,
            elapsed_s=round(time.time() - t_start, 1),
            pages=final_state.get("total_pages", 0),
            chars=len(full_text),
            ocr_method="vision",  # Always vision OCR (no native extraction)
        )
