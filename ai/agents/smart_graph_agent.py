"""
SmartDocumentAgent: memory-safe LangGraph-based document processor.

Graph topology:

    analyze_pdf
        |
        ├─ is_digital_text=True  →  native_text_extraction  →  extract_metadata  →  END
        │
        └─ is_digital_text=False →  batched_vision_ocr (loop until all pages done)
                                            └─ current_page >= total_pages → extract_metadata → END

Key design decisions:
- PyMuPDF pixmaps are del'd immediately after base64 encoding to avoid leaks.
- gc.collect() is called after each vision batch to release C-heap memory.
- LangGraph annotated state uses operator.add to accumulate text chunks without
  holding all images in memory simultaneously.
"""

import asyncio
import base64
import gc
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from datetime import datetime

import fitz  # PyMuPDF
import litellm
from pydantic import BaseModel, Field, field_validator

from agents.base import AgentResult, BaseDocumentAgent, DocumentMetadata
from agents.state import AgentState
from core.config import AgentConfig

log = logging.getLogger(__name__)

# Minimum characters per page to consider a page "digital text".
# NOTE: This threshold is currently unused — _analyze_pdf always routes to vision
# OCR regardless of native text content. Kept here as documentation of the
# original routing logic in case the fast path is ever re-enabled.
_DIGITAL_TEXT_THRESHOLD = 50

# Regex patterns for thinking tags emitted by some reasoning models
# (DeepSeek-R1, Qwen-QwQ, etc.) when thinking is not separated into
# reasoning_content by the provider/LiteLLM.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINKING_TAG_RE = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)


def _get_completion_text(response) -> str:
    """Return only the final answer text from a LiteLLM completion response.

    LiteLLM normalises reasoning providers so that:
      - ``message.reasoning_content`` holds the thinking trace (string).
      - ``message.content`` holds only the final answer.

    Two edge cases still need handling:

    1. **Block-list content** – Anthropic passes thinking as typed content
       blocks (``{"type": "thinking", ...}``).  When LiteLLM surfaces this as
       a list we filter to ``type == "text"`` blocks only.

    2. **Inline tags** – Older LiteLLM versions or models not yet normalised
       (DeepSeek-R1, Qwen-QwQ via Ollama, …) may prepend the thinking wrapped
       in ``<think>…</think>`` inside the content string.  We strip those.
    """
    message = response.choices[0].message
    content = message.content

    # Case 1: content is a list of typed blocks (Anthropic extended thinking).
    # Keep only text blocks; discard thinking blocks.
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "thinking")
        )

    content = content or ""

    # If LiteLLM already separated thinking into reasoning_content the content
    # string is clean — return it directly.
    if getattr(message, "reasoning_content", None):
        return content

    # Case 2: strip inline thinking tags for providers not yet normalised.
    content = _THINK_TAG_RE.sub("", content)
    content = _THINKING_TAG_RE.sub("", content)
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
    date: Optional[str] = Field(
        default=None, description="Primary document date as YYYY-MM-DD."
    )
    correspondent: Optional[str] = Field(
        default=None,
        description="Name of the issuing organisation — not the recipient.",
    )

    @field_validator("date", mode="before")
    @classmethod
    def strip_time(cls, v):
        if not isinstance(v, str):
            return v
        try:
            return datetime.fromisoformat(v).date().isoformat()
        except ValueError:
            return v


# ---------------------------------------------------------------------------
# Extraction Strategy Pattern: separate LLM extraction from orchestration
# ---------------------------------------------------------------------------


class BaseExtractionStrategy(ABC):
    """Abstract base for metadata extraction strategies."""

    def _fallback_parse(self, raw: str) -> dict:
        """Robust fallback parser for handling free-form or incomplete JSON output.

        For models that produce malformed JSON, extract key-value pairs directly
        without relying on JSON parsing. Looks for patterns like:
            "key": "value"
        and extracts them regardless of surrounding syntax errors.
        """
        try:
            return json.loads(raw)
        except Exception:
            pass

        # Fallback: extract key-value pairs directly using regex
        # Pattern: "key_name": value_content (handles string values with trailing commas, etc.)
        result = {}
        pattern = r'"([^"]+)"\s*:\s*"([^"]*)(?:"|,|"\s*,)'
        for match in re.finditer(pattern, raw):
            key, value = match.groups()
            result[key] = value

        if result:
            return result

        return {}

    @abstractmethod
    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        """Extract metadata from text using this strategy."""
        pass


class StructuredOutputStrategy(BaseExtractionStrategy):
    """Standard LLM extraction using JSON schema / response_format."""

    async def extract(self, text: str, config: AgentConfig) -> _ExtractedMetadata:
        """Use standard LLM with response_format tiering."""
        messages = [
            {"role": "system", "content": config.metadata_prompt},
            {"role": "user", "content": text},
        ]
        kwargs: dict = {
            "model": config.effective_metadata_model,
            "messages": messages,
            "num_retries": config.llm_retries,
            **config.get_metadata_litellm_kwargs(),
        }
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0

        # Tier response_format by what the model actually supports
        if litellm.supports_response_schema(model=config.effective_metadata_model):
            kwargs["response_format"] = _ExtractedMetadata
        elif "response_format" in (
            litellm.get_supported_openai_params(model=config.effective_metadata_model)
            or []
        ):
            kwargs["response_format"] = {"type": "json_object"}

        if config.metadata_api_base:
            kwargs["api_base"] = config.metadata_api_base

        response = await litellm.acompletion(**kwargs)
        raw = _get_completion_text(response) or "{}"
        log.info("Smart agent: metadata raw response: %s", raw)

        # Try strict Pydantic validation first, then fall back to loose parsing
        try:
            return _ExtractedMetadata.model_validate_json(raw)
        except Exception:
            data = self._fallback_parse(raw)
            return _ExtractedMetadata(
                title=data.get("title"),
                date=data.get("date"),
                correspondent=data.get("correspondent"),
            )


def _select_extraction_strategy(config: AgentConfig) -> BaseExtractionStrategy:
    """Select the appropriate extraction strategy based on the configured metadata model."""
    if "nuextract" in config.effective_metadata_model.lower():
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
        """Use NuExtract template passed via extra_body."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        template_str = json.dumps(json.loads(self._NUEXTRACT_TEMPLATE), indent=4)
        kwargs: dict = {
            "model": config.effective_metadata_model,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"template": template_str}},
            "num_retries": config.llm_retries,
            "temperature": 0,
        }

        if config.metadata_api_base:
            kwargs["api_base"] = config.metadata_api_base

        response = await litellm.acompletion(**kwargs)
        raw = _get_completion_text(response) or "{}"
        log.info("Smart agent: metadata raw response: %s", raw)

        # Parse using fallback chain and map custom keys back to standard schema
        data = self._fallback_parse(raw)
        return _ExtractedMetadata(
            title=data.get(self.document_title_key),
            date=data.get(self.date_key),
            correspondent=data.get(self.correspondent_key),
        )


# ---------------------------------------------------------------------------
# Graph node implementations (plain async functions — no class needed)
# ---------------------------------------------------------------------------


async def _analyze_pdf(state: AgentState, config: AgentConfig) -> dict:
    """Node 1: Measure page count and log native text density.

    Always routes to vision OCR (is_digital_text=False) regardless of how much
    native text the PDF contains. This ensures consistent extraction quality —
    native PyMuPDF text is unreliable for scanned documents and some digital PDFs
    where the embedded text is just metadata (e.g. a URL from a document library).
    """
    file_path = state["file_path"]
    doc = fitz.open(file_path)
    total_pages = len(doc)

    # Sample first 3 pages to log density (informational only — not used for routing)
    sample_pages = min(3, total_pages)
    total_chars = 0
    for i in range(sample_pages):
        total_chars += len(doc[i].get_text())
    doc.close()

    avg_chars_per_page = total_chars / sample_pages if sample_pages else 0

    log.info(
        "Smart agent: %d pages, avg %.0f chars/page (sampled %d) → vision OCR",
        total_pages,
        avg_chars_per_page,
        sample_pages,
    )

    return {
        "total_pages": total_pages,
        "is_digital_text": False,  # Always use vision OCR
        "current_page": 0,
        "extracted_text_chunks": [],
    }


async def _native_text_extraction(state: AgentState, config: AgentConfig) -> dict:
    """Node 2 (fast path): Extract text directly from a digital PDF using fitz.

    NOTE: This node is currently unreachable — _analyze_pdf always sets
    is_digital_text=False, so the graph always routes to _batched_vision_ocr.
    Kept in place in case native extraction is re-enabled in the future.
    """
    file_path = state["file_path"]
    doc = fitz.open(file_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks.append(text)
    doc.close()
    log.info("Smart agent: native text extraction — %d chunks", len(chunks))
    return {"extracted_text_chunks": chunks}


async def _batched_vision_ocr(state: AgentState, config: AgentConfig) -> dict:
    """Node 3: Render a batch of pages to base64 images and run vision OCR.

    This is the only active OCR path. Pages are processed in configurable batches
    (vision_batch_size) so only a few images are held in memory at once. Pixmaps
    are freed immediately after encoding to avoid C-heap accumulation.
    """
    file_path = state["file_path"]
    current_page = state["current_page"]
    batch_size = state["batch_size"]
    total_pages = state["total_pages"]
    language = state.get("language")

    end_page = min(current_page + batch_size, total_pages)
    log.info(
        "Smart agent: vision OCR pages %d–%d / %d",
        current_page + 1,
        end_page,
        total_pages,
    )

    doc = fitz.open(file_path)
    tasks = []

    for page_idx in range(current_page, end_page):
        page = doc[page_idx]
        # Cap render DPI so the longest image dimension stays within the model's
        # context budget (e.g. nanonets max-model-len=16128). Computing DPI
        # upfront avoids rendering at full resolution only to discard it.
        dpi = 300
        max_dim = config.ocr_max_image_dimension
        if max_dim:
            w_px = page.rect.width * dpi / 72
            h_px = page.rect.height * dpi / 72
            if max(w_px, h_px) > max_dim:
                dpi = int(dpi * max_dim / max(w_px, h_px))
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # Encode to PNG bytes and release the pixmap immediately
        png_bytes = pix.tobytes("png")
        b64 = base64.b64encode(png_bytes).decode()
        del pix, png_bytes  # release C-heap memory before async calls

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
            **config.get_ocr_litellm_kwargs(),
        }
        if config.ocr_api_base:
            kwargs["api_base"] = config.ocr_api_base

        tasks.append(litellm.acompletion(**kwargs))

    doc.close()

    responses = await asyncio.gather(*tasks)
    chunks = [_get_completion_text(r) for r in responses]

    # Force GC to reclaim any remaining image buffers from the event loop
    gc.collect()

    return {
        "extracted_text_chunks": chunks,
        "current_page": end_page,
    }


async def _extract_metadata(
    state: AgentState, config: AgentConfig, strategy: BaseExtractionStrategy
) -> dict:
    """Node 4: Join all text chunks and extract metadata using the provided strategy."""
    chunks = state["extracted_text_chunks"]
    full_text = "\n\n".join(chunks)

    # Truncate to first 4000 + last 2000 chars
    if len(full_text) > 6000:
        snippet = full_text[:4000] + "\n...\n" + full_text[-2000:]
    else:
        snippet = full_text

    # Use the strategy to extract metadata
    extracted = await strategy.extract(snippet, config)

    # Store final metadata back into state for the agent to read after graph completion
    return {"_extracted_metadata": extracted.model_dump(), "_full_text": full_text}


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

        # Bind config and strategy into each node via closures
        async def analyze_pdf(state: AgentState) -> dict:
            return await _analyze_pdf(state, config)

        async def native_text_extraction(state: AgentState) -> dict:
            return await _native_text_extraction(state, config)

        async def batched_vision_ocr(state: AgentState) -> dict:
            return await _batched_vision_ocr(state, config)

        async def extract_metadata(state: AgentState) -> dict:
            return await _extract_metadata(state, config, strategy)

        def route_after_analyze(state: AgentState) -> str:
            return (
                "native_text_extraction"
                if state["is_digital_text"]
                else "batched_vision_ocr"
            )

        def route_after_vision_ocr(state: AgentState) -> str:
            return (
                "batched_vision_ocr"
                if state["current_page"] < state["total_pages"]
                else "extract_metadata"
            )

        workflow = StateGraph(AgentState)
        workflow.add_node("analyze_pdf", analyze_pdf)
        workflow.add_node("native_text_extraction", native_text_extraction)
        workflow.add_node("batched_vision_ocr", batched_vision_ocr)
        workflow.add_node("extract_metadata", extract_metadata)

        workflow.set_entry_point("analyze_pdf")
        workflow.add_conditional_edges(
            "analyze_pdf",
            route_after_analyze,
            {
                "native_text_extraction": "native_text_extraction",
                "batched_vision_ocr": "batched_vision_ocr",
            },
        )
        workflow.add_edge("native_text_extraction", "extract_metadata")
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
        ocr_method = "native" if final_state.get("is_digital_text") else "vision"

        log.info(
            "Smart agent: done — title=%r date=%r correspondent=%r ocr=%s metadata=text",
            extracted_dict.get("title"),
            extracted_dict.get("date"),
            extracted_dict.get("correspondent"),
            ocr_method,
        )

        metadata = DocumentMetadata(
            title=extracted_dict.get("title"),
            document_date=extracted_dict.get("date"),
            correspondent=extracted_dict.get("correspondent"),
            full_ocr_transcript=full_text,
        )

        return AgentResult(
            metadata=metadata,
            elapsed_s=round(time.time() - t_start, 1),
            pages=final_state.get("total_pages", 0),
            chars=len(full_text),
            ocr_method=ocr_method,
        )
