"""
Legacy map-reduce pipeline agent (baseline).

Ports the original batch.py OCR logic:
  - Converts the full PDF to page images (300 DPI)
  - OCRs all pages in parallel, bounded by an asyncio semaphore
  - Extracts metadata with a text LLM using LiteLLM response_format

This serves as the performance baseline against SmartDocumentAgent.
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Optional

import fitz  # PyMuPDF
import litellm
from PIL import Image
from pydantic import BaseModel

from agents.base import AgentResult, BaseDocumentAgent, DocumentMetadata
from core.config import AgentConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal Pydantic model for LLM structured output (no transcript field)
# ---------------------------------------------------------------------------


class _ExtractedMetadata(BaseModel):
    title: Optional[str] = None
    document_date: Optional[str] = None
    correspondent: Optional[str] = None


# ---------------------------------------------------------------------------
# Page conversion helpers
# ---------------------------------------------------------------------------


def _document_to_pages(file_path: str) -> list[Image.Image]:
    """Convert a document file to a list of page images at 300 DPI."""
    # Try PDF via PyMuPDF
    try:
        doc = fitz.open(file_path)
        images = []
        for page in doc:
            mat = fitz.Matrix(300 / 72, 300 / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            del pix  # release pixmap memory immediately
        doc.close()
        if images:
            return images
    except Exception:
        pass

    # Fall back to PIL for image files (JPEG, PNG, TIFF, etc.)
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        img = Image.open(io.BytesIO(data))
        pages = []
        frame = 0
        while True:
            try:
                img.seek(frame)
                pages.append(img.copy().convert("RGB"))
                frame += 1
            except EOFError:
                break
        return pages if pages else [img.convert("RGB")]
    except Exception:
        pass

    raise ValueError(
        "Unsupported document format — only PDF and common image formats are supported"
    )


def _image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


async def _ocr_page(
    image: Image.Image,
    config: AgentConfig,
    language: str | None = None,
) -> str:
    """OCR a single page image using the configured vision LLM."""
    prompt = config.ocr_prompt
    if language:
        prompt = f"The document language is primarily '{language}'. " + prompt

    b64 = _image_to_b64(image)
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
    }
    if config.ocr_reasoning_effort:
        kwargs["reasoning_effort"] = config.ocr_reasoning_effort
    if config.ocr_api_base:
        kwargs["api_base"] = config.ocr_api_base

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content or ""


async def _extract_metadata(
    text: str,
    config: AgentConfig,
    existing: dict | None = None,
) -> _ExtractedMetadata:
    """Extract title, date, correspondent from document text using a text LLM."""
    # Take first 4000 + last 2000 chars so footer dates/signatures are included
    if len(text) > 6000:
        snippet = text[:4000] + "\n...\n" + text[-2000:]
    else:
        snippet = text

    # Build system prompt from config
    system_prompt = config.metadata_prompt
    if existing:
        hints = json.dumps(existing, ensure_ascii=False)
        system_prompt = (
            f"Existing metadata hints (use as context, may be wrong or missing):\n{hints}\n\n"
            + system_prompt
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": snippet},
    ]
    kwargs: dict = {
        "model": config.effective_metadata_model,
        "messages": messages,
        "temperature": 0,
        "response_format": _ExtractedMetadata,
        "num_retries": config.llm_retries,
    }
    if config.metadata_api_base:
        kwargs["api_base"] = config.metadata_api_base

    response = await litellm.acompletion(**kwargs)
    raw = response.choices[0].message.content or "{}"
    log.info("Metadata raw response: %s", raw)

    try:
        return _ExtractedMetadata.model_validate_json(raw)
    except Exception:
        # Some models return JSON wrapped in markdown fences
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return _ExtractedMetadata.model_validate_json(match.group())
            except Exception:
                pass
        # Manual parse fallback
        try:
            data = json.loads(match.group() if match else raw)
            return _ExtractedMetadata(
                title=data.get("title") or data.get("oneline_short_summary"),
                document_date=data.get("document_date") or data.get("date"),
                correspondent=data.get("correspondent")
                or data.get("correspondent_institution_or_individual"),
            )
        except Exception:
            return _ExtractedMetadata()


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class SeparatePipelineAgent(BaseDocumentAgent):
    """
    Baseline agent: convert entire PDF to images, OCR all pages in parallel.

    Memory footprint: all page images are held in RAM simultaneously.
    For documents > 100 pages this may be significant. Use SmartDocumentAgent
    for 500+ page PDFs.
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._semaphore: asyncio.Semaphore | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._config.ocr_concurrency)
        return self._semaphore

    async def process(self, file_path: str, existing_hints: dict) -> AgentResult:
        t_start = time.time()
        config = self._config

        # Convert document to page images
        try:
            images = _document_to_pages(file_path)
        except ValueError as e:
            log.warning("Legacy agent: page conversion failed: %s", e)
            raise
        except Exception as e:
            log.error("Legacy agent: page conversion error: %s", e)
            raise

        num_pages = len(images)
        if num_pages > 100:
            log.warning(
                "Legacy agent: %d pages — consider SmartDocumentAgent for memory safety",
                num_pages,
            )
        log.info(
            "Legacy agent: %d page(s), OCR via %s (concurrency=%d)",
            num_pages,
            config.ocr_model,
            config.ocr_concurrency,
        )

        # OCR all pages in parallel, bounded by semaphore
        language = existing_hints.get("language")
        sem = self._get_semaphore()

        async def _ocr_one(idx: int, img: Image.Image) -> tuple[int, str]:
            async with sem:
                text = await _ocr_page(img, config, language=language)
                log.debug("Legacy agent: page %d/%d — %d chars", idx, num_pages, len(text))
                return idx, text

        results = await asyncio.gather(*[
            _ocr_one(i, img) for i, img in enumerate(images, 1)
        ])
        page_texts = [text for _, text in sorted(results)]
        full_text = "\n\n".join(page_texts)
        log.info("Legacy agent: OCR complete — %d chars total", len(full_text))

        # Extract metadata
        try:
            extracted = await _extract_metadata(full_text, config, existing=existing_hints or None)
            log.info(
                "Legacy agent: metadata — title=%r date=%r correspondent=%r",
                extracted.title,
                extracted.document_date,
                extracted.correspondent,
            )
        except Exception as e:
            log.warning("Legacy agent: metadata extraction failed: %s", e)
            extracted = _ExtractedMetadata()

        metadata = DocumentMetadata(
            title=extracted.title,
            document_date=extracted.document_date,
            correspondent=extracted.correspondent,
            full_ocr_transcript=full_text,
        )

        return AgentResult(
            metadata=metadata,
            elapsed_s=round(time.time() - t_start, 1),
            pages=num_pages,
            chars=len(full_text),
            ocr_method="vision",
        )
