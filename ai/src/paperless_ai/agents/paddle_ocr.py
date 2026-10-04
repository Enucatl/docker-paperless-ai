"""Complete-document OCR using the PaddleOCR-VL parsing service."""

import asyncio
import base64
import hashlib
import html
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import niquests
from pypdf import PdfReader

from paperless_ai.core.config import AgentConfig


_IMAGE_FIELDS = {
    "image",
    "images",
    "inputImage",
    "outputImages",
    "markdownImages",
    "input_img",
    "output_img",
    "exports",
}
_HTML_IMAGE = re.compile(r"<img\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*>", re.IGNORECASE)
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\((?:[^()\n]|\([^()\n]*\))*\)")
_REFERENCE_IMAGE = re.compile(r"!\[([^\]]*)\](?:\[([^\]]*)\])?")
_REFERENCE_DEFINITION = re.compile(r"(?m)^[ \t]{0,3}\[([^\]]+)\]:[^\n]*$")
_IDENTITY_FIELDS = ("pipeline", "model", "layout_model")


async def fetch_ocr_metadata(config: AgentConfig) -> dict[str, str]:
    """Read and validate the parsing service's deployment identity."""
    if not config.ocr_endpoint:
        raise ValueError("Document OCR requires INFERENCE_OCR_ENDPOINT")
    url = config.ocr_endpoint.rstrip("/") + "/metadata"
    try:
        async with niquests.AsyncSession() as session:
            response = await session.get(url, timeout=5.0)
            response.raise_for_status()
            metadata = response.json()
    except (niquests.RequestException, ValueError) as exc:
        raise RuntimeError(f"OCR metadata request failed at {url}: {exc}") from exc
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(field), str) or not metadata[field].strip()
        for field in _IDENTITY_FIELDS
    ):
        raise ValueError(
            "OCR /metadata requires nonempty pipeline, model, and layout_model strings"
        )
    return {field: metadata[field] for field in _IDENTITY_FIELDS}


def _image_alt(match: re.Match) -> str:
    """Keep an HTML image's text alternative when discarding its source."""
    alt = re.search(
        r"\balt\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", match[0], re.IGNORECASE
    )
    return (
        html.unescape(next(value for value in alt.groups() if value is not None))
        if alt
        else ""
    )


def _without_images(text: str) -> str:
    """Remove image references while retaining captions and text whitespace."""
    references = set()

    def replace_reference(match: re.Match) -> str:
        """Preserve a reference image's caption and remember its definition."""
        references.add(" ".join((match[2] or match[1]).split()).casefold())
        return match[1]

    text = _HTML_IMAGE.sub(_image_alt, _MARKDOWN_IMAGE.sub(r"\1", text))
    text = _REFERENCE_IMAGE.sub(replace_reference, text)
    return _REFERENCE_DEFINITION.sub(
        lambda match: (
            "" if " ".join(match[1].split()).casefold() in references else match[0]
        ),
        text,
    )


def _prune_images(value: Any) -> Any:
    """Discard image/export fields while retaining structured recognition data."""
    if isinstance(value, dict):
        return {
            key: _prune_images(item)
            for key, item in value.items()
            if key not in _IMAGE_FIELDS
        }
    if isinstance(value, list):
        return [_prune_images(item) for item in value]
    if isinstance(value, str):
        return _without_images(value)
    return value


def _parse_response(payload: Any, page_count: int) -> tuple[str, dict, list[dict]]:
    """Validate page coverage and return image-free content and page records.

    Raises:
        ValueError: The service failed or returned an incomplete transcript.
    """
    if not isinstance(payload, dict) or type(payload.get("errorCode")) is not int:
        raise ValueError("Malformed Paddle API response")
    if payload["errorCode"] != 0:
        raise ValueError(
            f"Paddle API error {payload['errorCode']}: {payload.get('errorMsg', '')}"
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Missing Paddle result")
    info = result.get("dataInfo")
    if (
        not isinstance(info, dict)
        or info.get("type") != "pdf"
        or type(info.get("numPages")) is not int
        or info["numPages"] != page_count
        or not isinstance(info.get("pages"), list)
        or len(info["pages"]) != page_count
    ):
        raise ValueError("Paddle document information does not match source pages")
    results = result.get("layoutParsingResults")
    if not isinstance(results, list) or len(results) != page_count:
        raise ValueError("Paddle page coverage does not match source pages")
    pages = []
    # The official service strips page_index; its result array is in PDF order.
    for index, page in enumerate(results):
        if not isinstance(page, dict):
            raise ValueError(f"Malformed Paddle page {index}")
        pruned = page.get("prunedResult")
        markdown = page.get("markdown")
        if (
            not isinstance(pruned, dict)
            or not isinstance(pruned.get("parsing_res_list"), list)
            or any(not isinstance(block, dict) for block in pruned["parsing_res_list"])
            or not isinstance(markdown, dict)
            or not isinstance(markdown.get("text"), str)
        ):
            raise ValueError(f"Malformed Paddle page {index}")
        pages.append(
            {
                "page_index": index,
                "prunedResult": _prune_images(pruned),
                "markdown": {"text": _without_images(markdown["text"])},
            }
        )
    text = "\n\n".join(page["markdown"]["text"] for page in pages).strip()
    # Empty HTML wrappers left after image removal are not recognized text.
    if not re.sub(r"<[^>]+>", "", text).strip():
        raise ValueError("Paddle returned an empty transcript")
    return text, _prune_images(info), pages


async def run_paddle_ocr(
    file_path: str, config: AgentConfig, metadata: dict[str, str] | None = None
) -> tuple[str, dict, int, float]:
    """Parse a complete PDF and return Markdown, provenance, pages and seconds.

    Args:
        file_path: Local PDF to send without rendering or sampling pages.
        config: Parsing service base URL and request timeout.
        metadata: Validated batch identity, or None for a standalone job.

    Returns:
        Ordered Markdown, structured output, source page count and elapsed time.

    Raises:
        ValueError: The PDF or response cannot provide a complete transcript.
        niquests.RequestException: The service request fails.
    """
    if not config.ocr_endpoint:
        raise ValueError("Document OCR requires INFERENCE_OCR_ENDPOINT")
    if metadata is None:
        metadata = await fetch_ocr_metadata(config)
    started = time.monotonic()
    source = await asyncio.to_thread(Path(file_path).read_bytes)
    page_count = len(PdfReader(BytesIO(source)).pages)
    if page_count == 0:
        raise ValueError("Cannot OCR a PDF without pages")
    async with niquests.AsyncSession() as session:
        response = await session.post(
            config.ocr_endpoint.rstrip("/") + "/layout-parsing",
            json={
                "file": base64.b64encode(source).decode("ascii"),
                "fileType": 0,
                "useLayoutDetection": True,
                "markdownIgnoreLabels": [],
                "returnMarkdownImages": False,
                "visualize": False,
                "restructurePages": False,
            },
            timeout=config.ocr_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text, info, pages = _parse_response(payload, page_count)
    elapsed = time.monotonic() - started
    output = {
        "schema_version": 1,
        **metadata,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "page_count": page_count,
        "elapsed_seconds": elapsed,
        "dataInfo": info,
        "pages": pages,
    }
    return text, output, page_count, elapsed
