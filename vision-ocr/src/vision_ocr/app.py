"""Serve complete PDFs using a sequential OpenAI-compatible vision pipeline."""

import asyncio
import base64
import binascii
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import fitz
import niquests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from shared_inference import InferenceClient
from shared_inference.errors import InferenceError, InferenceTimeoutError


class Settings(BaseSettings):
    """Service-specific connection, rendering and generation configuration."""

    model_config = SettingsConfigDict(env_prefix="VISION_OCR_")
    endpoint: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key: str | None = None
    prompt: str = Field(
        default_factory=lambda: Path(__file__).with_name("prompt.txt").read_text()
    )
    max_image_dimension: int | None = Field(default=None, gt=0)
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float | None = None
    reasoning_effort: str | None = "minimal"
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=600, gt=0)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Require an HTTP OpenAI-compatible API base URL."""
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("endpoint must be an HTTP(S) API base URL")
        return value.rstrip("/")

    @field_validator("extra_kwargs")
    @classmethod
    def validate_extra_kwargs(cls, value: dict) -> dict:
        """Prevent generation extras from overriding the request contract."""
        if {"model", "messages", "stream", "n", "domain"} & value.keys():
            raise ValueError(
                "extra_kwargs cannot change model, messages or response mode"
            )
        return value


class ParseRequest(BaseModel):
    """Supported subset of Paddle's base64 PDF request contract."""

    model_config = ConfigDict(extra="forbid", strict=True)
    file: str = Field(min_length=1)
    fileType: Literal[0] = 0
    useLayoutDetection: bool = True
    markdownIgnoreLabels: list[str] = Field(default_factory=list, max_length=0)
    returnMarkdownImages: Literal[False] = False
    visualize: Literal[False] = False
    restructurePages: Literal[False] = False


def render_page(page: fitz.Page, max_dimension: int | None) -> str:
    """Render a page at 300 DPI, bounded by the configured longest dimension."""
    scale = 300 / 72
    if max_dimension:
        scale = min(scale, max_dimension / max(page.rect.width, page.rect.height))
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return base64.b64encode(pixmap.tobytes("png")).decode("ascii")


def completion_text(response: Any) -> str:
    """Extract transcription while rejecting incomplete or malformed responses."""
    choices = response.raw.get("choices")
    if not choices or choices[0].get("finish_reason") not in {None, "stop"}:
        raise ValueError("Upstream returned an incomplete completion")
    if response.message.get("refusal"):
        raise ValueError("Upstream refused transcription")
    content = response.message.get("content")
    if isinstance(content, list):
        if any(
            not isinstance(block, dict) or block.get("type") not in {"text", "thinking"}
            for block in content
        ):
            raise ValueError("Upstream returned unsupported transcription blocks")
        if not any(block["type"] == "text" for block in content):
            raise ValueError("Upstream returned no transcription blocks")
        content = "".join(
            block["text"] for block in content if block.get("type") == "text"
        )
    if not isinstance(content, str):
        raise ValueError("Upstream returned no transcription")
    return re.sub(
        r"<(think|thinking)>.*?</\1>", "", content, flags=re.DOTALL | re.IGNORECASE
    ).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one inference client and release its connections on shutdown."""
    settings = Settings()
    app.state.settings = settings
    app.state.lock = asyncio.Lock()
    app.state.client = InferenceClient(
        base_url=settings.endpoint,
        api_key=settings.api_key,
        timeout=settings.timeout,
        domain="vision_ocr",
        provider=(
            "openrouter"
            if urlsplit(settings.endpoint).hostname == "openrouter.ai"
            else "openai-compatible"
        ),
    )
    try:
        yield
    finally:
        await app.state.client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    """Check upstream reachability without performing inference."""
    settings = app.state.settings
    headers = (
        {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    )
    try:
        async with asyncio.timeout(3):
            response = await app.state.client.get(
                f"{settings.endpoint}/models", headers=headers, timeout=3
            )
            response.raise_for_status()
    except (niquests.RequestException, TimeoutError, InferenceError) as exc:
        raise HTTPException(503, "Vision upstream unavailable") from exc
    return {"status": "ok"}


@app.get("/metadata")
async def metadata(response: Response) -> dict[str, str]:
    """Report the configured vision model and absence of a layout model."""
    response.headers["Cache-Control"] = "no-store"
    return {
        "pipeline": "vision-ocr",
        "model": app.state.settings.model,
        "layout_model": "none",
    }


async def parse_document(request: ParseRequest, deadline: float) -> dict:
    """Render and transcribe all PDF pages before returning any page results."""
    try:
        source = base64.b64decode(request.file, validate=True)
        document = fitz.open(stream=source, filetype="pdf")
    except (ValueError, binascii.Error, fitz.FileDataError) as exc:
        raise HTTPException(400, "Invalid base64 PDF") from exc
    with document:
        if not document.is_pdf or document.needs_pass or not document.page_count:
            raise HTTPException(400, "Expected an unencrypted PDF with pages")
        settings = app.state.settings
        kwargs = {"max_tokens": settings.max_tokens}
        if settings.temperature is not None:
            kwargs["temperature"] = settings.temperature
        if settings.reasoning_effort:
            kwargs["reasoning_effort"] = settings.reasoning_effort
        kwargs.update(settings.extra_kwargs)
        pages, results = [], []
        models = set()
        for page in document:
            image = render_page(page, settings.max_image_dimension)
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("PDF rendering exceeded the document deadline")
            await asyncio.sleep(0)
            response = await app.state.client.complete(
                model=settings.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                            {"type": "text", "text": settings.prompt},
                        ],
                    }
                ],
                **kwargs,
            )
            text = completion_text(response)
            models.add(
                response.raw.get("model")
                or (
                    settings.model
                    if urlsplit(settings.endpoint).hostname == "openrouter.ai"
                    else settings.model.removeprefix("openai/")
                )
            )
            del image, response
            pages.append({"width": page.rect.width, "height": page.rect.height})
            results.append(
                {
                    "prunedResult": {
                        "parsing_res_list": (
                            [{"block_label": "text", "block_content": text}]
                            if text
                            else []
                        )
                    },
                    "markdown": {"text": text},
                }
            )
        if len(models) != 1:
            raise ValueError("Upstream changed model identity within a document")
        return {
            "errorCode": 0,
            "errorMsg": "Success",
            "result": {
                "dataInfo": {"type": "pdf", "numPages": len(pages), "pages": pages},
                "layoutParsingResults": results,
            },
        }


@app.post("/layout-parsing")
async def layout_parsing(request: ParseRequest) -> dict:
    """Serialize documents and fail atomically on timeout or upstream errors."""
    try:
        deadline = asyncio.get_running_loop().time() + app.state.settings.timeout
        async with asyncio.timeout_at(deadline):
            # ponytail: one document at a time; add workers if throughput requires it.
            async with app.state.lock:
                result = await parse_document(request, deadline)
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Vision OCR exceeded the document deadline")
                return result
    except (TimeoutError, InferenceTimeoutError) as exc:
        raise HTTPException(504, "Vision OCR timed out") from exc
    except (InferenceError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            502, "Vision OCR failed to produce a complete document"
        ) from exc
