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
import io
import json
import logging
import re
import time
from typing import Optional

import fitz  # PyMuPDF
import litellm
from pydantic import BaseModel

from agents.base import AgentResult, BaseDocumentAgent, DocumentMetadata
from agents.state import AgentState
from core.config import AgentConfig

log = logging.getLogger(__name__)

# Minimum characters per page to consider a page "digital text"
_DIGITAL_TEXT_THRESHOLD = 50


class _ExtractedMetadata(BaseModel):
    title: Optional[str] = None
    document_date: Optional[str] = None
    correspondent: Optional[str] = None


# ---------------------------------------------------------------------------
# Graph node implementations (plain async functions — no class needed)
# ---------------------------------------------------------------------------


async def _analyze_pdf(state: AgentState, config: AgentConfig) -> dict:
    """Node 1: Detect page count and whether the PDF contains native digital text."""
    file_path = state["file_path"]
    doc = fitz.open(file_path)
    total_pages = len(doc)

    # Sample first 3 pages for native text
    sample_pages = min(3, total_pages)
    total_chars = 0
    for i in range(sample_pages):
        total_chars += len(doc[i].get_text())
    doc.close()

    avg_chars_per_page = total_chars / sample_pages if sample_pages else 0
    is_digital_text = avg_chars_per_page >= _DIGITAL_TEXT_THRESHOLD

    log.info(
        "Smart agent: %d pages, avg %.0f chars/page (sampled %d) → %s",
        total_pages,
        avg_chars_per_page,
        sample_pages,
        "native text" if is_digital_text else "vision OCR",
    )

    return {
        "total_pages": total_pages,
        "is_digital_text": is_digital_text,
        "current_page": 0,
        "extracted_text_chunks": [],
    }


async def _native_text_extraction(state: AgentState, config: AgentConfig) -> dict:
    """Node 2 (fast path): Extract text directly from a digital PDF using fitz."""
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
    """Node 3 (slow path): Render a batch of pages to base64 and run vision OCR."""
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
        mat = fitz.Matrix(300 / 72, 300 / 72)
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
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        kwargs: dict = {
            "model": config.ocr_model,
            "messages": messages,
            "num_retries": config.llm_retries,
            **config.get_litellm_kwargs(),
        }
        if config.ocr_api_base:
            kwargs["api_base"] = config.ocr_api_base

        tasks.append(litellm.acompletion(**kwargs))

    doc.close()

    responses = await asyncio.gather(*tasks)
    chunks = [r.choices[0].message.content or "" for r in responses]

    # Force GC to reclaim any remaining image buffers from the event loop
    gc.collect()

    return {
        "extracted_text_chunks": chunks,
        "current_page": end_page,
    }


async def _extract_metadata(state: AgentState, config: AgentConfig) -> dict:
    """Node 4: Join all text chunks and call the text LLM for structured metadata."""
    chunks = state["extracted_text_chunks"]
    full_text = "\n\n".join(chunks)

    # Truncate to first 4000 + last 2000 chars
    if len(full_text) > 6000:
        snippet = full_text[:4000] + "\n...\n" + full_text[-2000:]
    else:
        snippet = full_text

    messages = [
        {"role": "system", "content": config.metadata_prompt},
        {"role": "user", "content": snippet},
    ]
    kwargs: dict = {
        "model": config.effective_metadata_model,
        "messages": messages,
        "response_format": _ExtractedMetadata,
        "num_retries": config.llm_retries,
        **config.get_litellm_kwargs(),
    }
    # For metadata extraction, we usually want low temperature, but we allow 
    # the experiment to override it if set in get_litellm_kwargs.
    if "temperature" not in kwargs:
        kwargs["temperature"] = 0

    if config.metadata_api_base:
        kwargs["api_base"] = config.metadata_api_base

    response = await litellm.acompletion(**kwargs)
    raw = response.choices[0].message.content or "{}"
    log.info("Smart agent: metadata raw response: %s", raw)

    try:
        extracted = _ExtractedMetadata.model_validate_json(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            extracted = _ExtractedMetadata.model_validate_json(match.group() if match else "{}")
        except Exception:
            try:
                data = json.loads(match.group() if match else raw)
                extracted = _ExtractedMetadata(
                    title=data.get("title") or data.get("oneline_short_summary"),
                    document_date=data.get("document_date") or data.get("date"),
                    correspondent=data.get("correspondent")
                    or data.get("correspondent_institution_or_individual"),
                )
            except Exception:
                extracted = _ExtractedMetadata()

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

    def __init__(self, config: AgentConfig):
        self._config = config
        self._graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as e:
            raise ImportError(
                "langgraph is required for SmartDocumentAgent: pip install langgraph"
            ) from e

        config = self._config

        # Bind config into each node via closures
        async def analyze_pdf(state: AgentState) -> dict:
            return await _analyze_pdf(state, config)

        async def native_text_extraction(state: AgentState) -> dict:
            return await _native_text_extraction(state, config)

        async def batched_vision_ocr(state: AgentState) -> dict:
            return await _batched_vision_ocr(state, config)

        async def extract_metadata(state: AgentState) -> dict:
            return await _extract_metadata(state, config)

        def route_after_analyze(state: AgentState) -> str:
            return "native_text_extraction" if state["is_digital_text"] else "batched_vision_ocr"

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
        full_text = final_state.get("_full_text", "\n\n".join(final_state.get("extracted_text_chunks", [])))
        ocr_method = "native" if final_state.get("is_digital_text") else "vision"

        log.info(
            "Smart agent: done — title=%r date=%r correspondent=%r method=%s",
            extracted_dict.get("title"),
            extracted_dict.get("document_date"),
            extracted_dict.get("correspondent"),
            ocr_method,
        )

        metadata = DocumentMetadata(
            title=extracted_dict.get("title"),
            document_date=extracted_dict.get("document_date"),
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
