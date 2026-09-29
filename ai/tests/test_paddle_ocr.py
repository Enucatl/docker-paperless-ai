"""Mocked parsing-service contract checks; no GPU or Paperless required."""

import base64
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import fitz
import niquests
import pytest

from paperless_ai.agents.paddle_ocr import (
    _parse_response,
    _without_images,
    fetch_ocr_metadata,
    run_paddle_ocr,
)
from paperless_ai.core.config import AgentConfig

_METADATA = {
    "pipeline": "PaddleOCR-VL-1.6",
    "model": "PaddlePaddle/PaddleOCR-VL-1.6",
    "layout_model": "PP-DocLayoutV3",
}


def _response(*texts: str) -> dict:
    """Build the official service's per-page response shape."""
    return {
        "errorCode": 0,
        "result": {
            "dataInfo": {
                "type": "pdf",
                "numPages": len(texts),
                "pages": [{"width": 100, "height": 200} for _ in texts],
            },
            "layoutParsingResults": [
                {
                    "prunedResult": {
                        "parsing_res_list": [
                            {"block_label": "text", "block_content": text}
                        ]
                        if text
                        else [],
                    },
                    "markdown": {"text": text},
                }
                for text in texts
            ],
        },
    }


def test_page_order_blank_pages_and_image_omission() -> None:
    """Keep text, tables and equations in order while discarding image payloads."""
    payload = _response(
        '# Résumé 日本語\n![Caption](images/figure(1).png)\n<img src="data:image/png;base64,secret">',
        "",
        "| A | B |\n|---|---|\n| 1 | 2 |\n$$x^2$$\n<table><tr><td>Text</td></tr></table>",
    )
    first = payload["result"]["layoutParsingResults"][0]
    first["markdown"]["images"] = {"image.png": "secret"}
    first["outputImages"] = {"preview": "secret"}
    first["inputImage"] = "secret"
    first["exports"] = {"docx": {"content": "secret"}}
    first["prunedResult"]["parsing_res_list"][0]["image"] = "secret"
    first["prunedResult"]["doc_preprocessor_res"] = {"output_img": "secret", "angle": 0}
    original = copy.deepcopy(payload)
    text, info, pages = _parse_response(payload, 3)
    assert text.startswith("# Résumé 日本語\nCaption")
    assert "| 1 | 2 |" in text and "$$x^2$$" in text
    assert "<td>Text</td>" in text
    assert [page["page_index"] for page in pages] == [0, 1, 2]
    assert pages[1]["markdown"]["text"] == ""
    assert info["numPages"] == 3
    stored = json.dumps(pages)
    assert "secret" not in stored and "images/" not in stored
    assert "<img" not in stored and "exports" not in stored
    assert payload == original


def test_image_captions_references_and_whitespace() -> None:
    """Keep captions and ordinary text exactly while removing image references."""
    original = '  indented text  \n<img alt="A &amp; B" src="secret">\n![Caption][figure]\n![other][]\n[figure]: data:image/png;base64,secret\n[other]: secret\n[link]: https://example.com'
    cleaned = _without_images(original)
    assert cleaned.startswith("  indented text  \nA & B\nCaption\nother\n")
    assert "secret" not in cleaned
    assert "[link]: https://example.com" in cleaned


@pytest.mark.parametrize("count", [1, 41, 75])
def test_complete_page_coverage(count: int) -> None:
    """Preserve every page, including PDFs over the previous sampling limit."""
    text, _, pages = _parse_response(_response(*(str(i) for i in range(count))), count)
    assert text.split("\n\n") == [str(i) for i in range(count)]
    assert len(pages) == count


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"errorCode": False},
        {"errorCode": 500, "errorMsg": "failed"},
        {"errorCode": 0},
        _response(""),
        _response(' <div><img src="image.png"></div> '),
    ],
)
def test_reject_api_errors_and_empty_transcripts(payload: object) -> None:
    """Errors and wholly empty documents cannot replace stored transcripts."""
    with pytest.raises(ValueError):
        _parse_response(payload, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_page",
        "extra_page",
        "wrong_count",
        "missing_info",
        "missing_markdown",
        "invalid_text",
        "invalid_pruned",
        "invalid_block",
        "invalid_page",
    ],
)
def test_reject_incomplete_or_malformed_pages(mutation: str) -> None:
    """Count agreement alone cannot make malformed page data valid."""
    payload = _response("first", "second")
    result = payload["result"]
    pages = result["layoutParsingResults"]
    if mutation == "missing_page":
        pages.pop()
    elif mutation == "extra_page":
        pages.append(copy.deepcopy(pages[0]))
    elif mutation == "wrong_count":
        result["dataInfo"]["numPages"] = 1
    elif mutation == "missing_info":
        result.pop("dataInfo")
    elif mutation == "missing_markdown":
        pages[0].pop("markdown")
    elif mutation == "invalid_text":
        pages[0]["markdown"]["text"] = None
    elif mutation == "invalid_pruned":
        pages[0]["prunedResult"] = {}
    elif mutation == "invalid_block":
        pages[0]["prunedResult"]["parsing_res_list"] = [None]
    else:
        pages[0] = None
    with pytest.raises(ValueError):
        _parse_response(payload, 2)


async def test_complete_pdf_request_and_metadata(tmp_path: Path) -> None:
    """Fetch identity for a standalone job and parse the unmodified response."""
    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        doc.save(path)
    config = AgentConfig(
        metadata_model="metadata",
        chat_model="chat",
        ocr_endpoint="http://paddle:8080/",
        ocr_timeout=123,
    )
    response = Mock()
    response.json.return_value = _response("one", "two")
    metadata_response = Mock()
    metadata_response.json.return_value = {**_METADATA, "extra": "allowed"}
    with (
        patch(
            "niquests.AsyncSession.get",
            new_callable=AsyncMock,
            return_value=metadata_response,
        ) as get,
        patch(
            "niquests.AsyncSession.post", new_callable=AsyncMock, return_value=response
        ) as post,
    ):
        text, output, count, elapsed = await run_paddle_ocr(str(path), config)
    get.assert_awaited_once_with("http://paddle:8080/metadata", timeout=5.0)
    post.assert_awaited_once()
    assert post.call_args.args == ("http://paddle:8080/layout-parsing",)
    assert post.call_args.kwargs["timeout"] == 123
    body = post.call_args.kwargs["json"]
    assert base64.b64decode(body.pop("file")) == path.read_bytes()
    assert body == {
        "fileType": 0,
        "useLayoutDetection": True,
        "markdownIgnoreLabels": [],
        "returnMarkdownImages": False,
        "visualize": False,
        "restructurePages": False,
    }
    response.raise_for_status.assert_called_once()
    assert (text, count) == ("one\n\ntwo", 2)
    assert elapsed >= 0
    assert output["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert output["schema_version"] == 1 and output["timestamp"].endswith("+00:00")
    for key, value in _METADATA.items():
        assert output[key] == value
    assert "extra" not in output
    assert output["page_count"] == count


@pytest.mark.parametrize(
    "error",
    [
        niquests.HTTPError("503"),
        niquests.Timeout("timeout"),
        ValueError("invalid JSON"),
    ],
)
async def test_request_failures_propagate_for_worker_retry(
    tmp_path: Path, error: Exception
) -> None:
    """Let the existing worker retry mechanism handle HTTP and parsing failures."""
    path = tmp_path / "source.pdf"
    with fitz.open() as doc:
        doc.new_page()
        doc.save(path)
    config = AgentConfig(
        metadata_model="metadata", chat_model="chat", ocr_endpoint="http://paddle"
    )
    response = Mock()
    if isinstance(error, niquests.HTTPError):
        response.raise_for_status.side_effect = error
    else:
        response.json.side_effect = error
    with patch(
        "niquests.AsyncSession.post", new_callable=AsyncMock, return_value=response
    ) as post:
        if isinstance(error, niquests.Timeout):
            post.side_effect = error
        with pytest.raises(type(error)):
            await run_paddle_ocr(str(path), config, _METADATA)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        {},
        {"pipeline": "x", "model": "y"},
        {**_METADATA, "model": "  "},
        {**_METADATA, "layout_model": 7},
    ],
)
async def test_reject_missing_or_malformed_metadata(metadata) -> None:
    """Reject missing or invalid model identity before parsing starts."""
    response = Mock()
    response.json.return_value = metadata
    config = AgentConfig(
        metadata_model="metadata", chat_model="chat", ocr_endpoint="http://ocr"
    )
    with patch("niquests.AsyncSession.get", AsyncMock(return_value=response)):
        with pytest.raises(ValueError, match="OCR /metadata requires nonempty"):
            await fetch_ocr_metadata(config)


@pytest.mark.parametrize(
    "error",
    [
        niquests.HTTPError("503"),
        niquests.Timeout("timeout"),
        ValueError("invalid JSON"),
    ],
)
async def test_metadata_http_and_json_failures(error: Exception) -> None:
    """Surface retrieval failures clearly without returning an old identity."""
    response = Mock()
    if isinstance(error, niquests.HTTPError):
        response.raise_for_status.side_effect = error
    else:
        response.json.side_effect = error
    config = AgentConfig(
        metadata_model="metadata", chat_model="chat", ocr_endpoint="http://ocr"
    )
    with patch("niquests.AsyncSession.get", AsyncMock(return_value=response)) as get:
        if isinstance(error, niquests.Timeout):
            get.side_effect = error
        with pytest.raises(RuntimeError, match="OCR metadata request failed"):
            await fetch_ocr_metadata(config)
