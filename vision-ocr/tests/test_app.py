"""Exercise the HTTP adapter with mocked inference and real PDF rendering."""

import base64
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import fitz
import pytest
from fastapi.testclient import TestClient
from shared_inference.errors import InferenceError, InferenceTimeoutError

from vision_ocr.app import app
from vision_ocr.app import ParseRequest, layout_parsing


def pdf_request(pages: int = 3) -> dict:
    """Build the exact PDF mode used by Paperless AI."""
    with fitz.open() as document:
        for _ in range(pages):
            document.new_page(width=72, height=72)
        return {
            "file": base64.b64encode(document.tobytes()).decode(),
            "fileType": 0,
            "useLayoutDetection": True,
            "markdownIgnoreLabels": [],
            "returnMarkdownImages": False,
            "visualize": False,
            "restructurePages": False,
        }


def completion(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    """Return the shared client's completion shape."""
    return SimpleNamespace(
        message={"content": text},
        raw={"model": "actual-model", "choices": [{"finish_reason": finish_reason}]},
    )


@pytest.fixture
def client(monkeypatch):
    """Start the service with one mocked lifespan client."""
    monkeypatch.setenv("VISION_OCR_ENDPOINT", "https://inference.invalid/v1")
    monkeypatch.setenv("VISION_OCR_MODEL", "configured-model")
    monkeypatch.setenv("VISION_OCR_API_KEY", "test-key")
    monkeypatch.setenv("VISION_OCR_MAX_IMAGE_DIMENSION", "64")
    inference = SimpleNamespace(
        complete=AsyncMock(), get=AsyncMock(), close=AsyncMock()
    )
    monkeypatch.setattr("vision_ocr.app.InferenceClient", lambda **_: inference)
    with TestClient(app) as client:
        yield client, inference
    inference.close.assert_awaited_once()


def test_complete_pdf_preserves_pages_and_truthful_identity(client):
    """Preserve Unicode, tables and blank pages without invented geometry."""
    http, inference = client
    inference.complete.side_effect = [
        completion("<think>private</think>Résumé 日本語"),
        completion(""),
        completion("<table><tr><td>42 €</td></tr></table>"),
    ]
    response = http.post("/layout-parsing", json=pdf_request())
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result["dataInfo"]["numPages"] == 3
    assert "provenance" not in result
    metadata = http.get("/metadata")
    assert metadata.status_code == 200
    assert metadata.headers["Cache-Control"] == "no-store"
    assert metadata.json() == {
        "pipeline": "vision-ocr",
        "model": "configured-model",
        "layout_model": "none",
    }
    pages = result["layoutParsingResults"]
    assert [page["markdown"]["text"] for page in pages] == [
        "Résumé 日本語",
        "",
        "<table><tr><td>42 €</td></tr></table>",
    ]
    assert pages[1]["prunedResult"]["parsing_res_list"] == []
    assert set(pages[0]["prunedResult"]["parsing_res_list"][0]) == {
        "block_label",
        "block_content",
    }
    assert inference.complete.await_count == 3
    url = inference.complete.call_args.kwargs["messages"][0]["content"][0]["image_url"][
        "url"
    ]
    pixmap = fitz.Pixmap(base64.b64decode(url.split(",", 1)[1]))
    assert max(pixmap.width, pixmap.height) == 64


def test_actual_paperless_parser_accepts_adapter(client, monkeypatch, tmp_path):
    """Validate adapter output with the consumer's real parsing implementation."""
    parser_path = Path(__file__).parents[2] / "ai/src/paperless_ai/agents/paddle_ocr.py"
    spec = importlib.util.spec_from_file_location("contract_parser", parser_path)
    parser = importlib.util.module_from_spec(spec)
    # Only the config annotation requires the application dependency tree.
    # Use the existing AI environment for this cross-package check.
    if importlib.util.find_spec("paperless_ai") is None:
        pytest.skip("Run with the AI environment to verify the cross-package contract")
    spec.loader.exec_module(parser)
    http, inference = client
    table = "<table><tr><td>42 €</td></tr></table>"
    inference.complete.side_effect = [
        completion("Résumé 日本語"),
        completion(""),
        completion(table),
    ]
    source = tmp_path / "source.pdf"
    source.write_bytes(base64.b64decode(pdf_request(3)["file"]))
    session = AsyncMock()

    async def post(url, *, json, timeout):
        """Send the consumer's real outbound payload to the adapter route."""
        assert url == "http://vision-ocr:8000/layout-parsing"
        assert timeout == 600
        return http.post("/layout-parsing", json=json)

    session.post.side_effect = post

    async def get(url, *, timeout):
        """Return the adapter's static deployment identity."""
        assert url == "http://vision-ocr:8000/metadata"
        assert timeout == 5.0
        return http.get("/metadata")

    session.get.side_effect = get
    session.__aenter__.return_value = session
    monkeypatch.setattr(parser.niquests, "AsyncSession", lambda: session)
    text, output, count, _ = asyncio.run(
        parser.run_paddle_ocr(
            str(source),
            SimpleNamespace(ocr_endpoint="http://vision-ocr:8000", ocr_timeout=600),
        )
    )
    assert text == f"Résumé 日本語\n\n\n\n{table}"
    assert output["dataInfo"]["numPages"] == len(output["pages"]) == count == 3
    assert output["pages"][1]["markdown"]["text"] == ""
    assert output["pipeline"] == "vision-ocr"
    assert output["model"] == "configured-model"
    assert output["layout_model"] == "none"


@pytest.mark.parametrize("file", ["not base64", base64.b64encode(b"not PDF").decode()])
def test_invalid_pdf(client, file):
    """Reject invalid inputs before inference."""
    http, inference = client
    assert http.post("/layout-parsing", json={"file": file}).status_code == 400
    inference.complete.assert_not_called()


@pytest.mark.parametrize(
    "mode",
    [
        {"fileType": 1},
        {"returnMarkdownImages": True},
        {"visualize": True},
        {"restructurePages": True},
        {"markdownIgnoreLabels": ["text"]},
        {"pages": [0]},
    ],
)
def test_unsupported_modes(client, mode):
    """Reject modes that cannot satisfy the adapter contract."""
    http, inference = client
    assert http.post("/layout-parsing", json=pdf_request() | mode).status_code == 422
    inference.complete.assert_not_called()


@pytest.mark.parametrize(
    "failure, status",
    [
        (InferenceError("unavailable"), 502),
        (InferenceTimeoutError("timeout"), 504),
        (completion("partial", "length"), 502),
        (completion("filtered", "content_filter"), 502),
    ],
)
def test_late_failure_never_returns_partial_document(client, failure, status):
    """Discard previous successful pages when a later page fails."""
    http, inference = client
    inference.complete.side_effect = [completion("first page"), failure]
    response = http.post("/layout-parsing", json=pdf_request())
    assert response.status_code == status
    assert "result" not in response.json()
    assert inference.complete.await_count == 2


@pytest.mark.parametrize(
    "message",
    [
        {"content": "", "refusal": "cannot transcribe"},
        {"content": [{"type": "refusal", "refusal": "cannot transcribe"}]},
        {"content": [{"type": "image", "url": "https://example.invalid/image"}]},
        {"content": [{"type": "thinking", "thinking": "reasoning only"}]},
        {"content": []},
    ],
)
def test_late_refusal_never_becomes_blank_page(client, message):
    """Refusals and unsupported content fail the complete document."""
    http, inference = client
    refused = completion("")
    refused.message = message
    inference.complete.side_effect = [completion("first page"), refused]
    response = http.post("/layout-parsing", json=pdf_request())
    assert response.status_code == 502
    assert "result" not in response.json()
    assert inference.complete.await_count == 2


def test_health_only_queries_models(client):
    """Health checks send authentication and a short timeout without inference."""
    http, inference = client
    assert http.get("/metadata").json()["model"] == "configured-model"
    inference.get.assert_not_awaited()
    inference.complete.assert_not_called()
    inference.get.return_value = SimpleNamespace(raise_for_status=lambda: None)
    assert http.get("/health").status_code == 200
    inference.get.assert_awaited_once_with(
        "https://inference.invalid/v1/models",
        headers={"Authorization": "Bearer test-key"},
        timeout=3,
    )
    inference.complete.assert_not_called()
    inference.get.side_effect = TimeoutError()
    assert http.get("/health").status_code == 503


def test_documents_are_serialized(client):
    """A second document cannot start inference while the first is running."""
    _, inference = client

    async def run():
        """Hold the first completion while scheduling the next document."""
        entered, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def complete(**kwargs):
            """Pause the first document at its first page."""
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await release.wait()
            return completion("text")

        app.state.lock = asyncio.Lock()
        inference.complete.side_effect = complete
        request = ParseRequest(**pdf_request(2))
        first = asyncio.create_task(layout_parsing(request))
        await entered.wait()
        second = asyncio.create_task(layout_parsing(request))
        await asyncio.sleep(0)
        assert calls == 1
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert calls == 4
        assert first_result["errorCode"] == second_result["errorCode"] == 0

    asyncio.run(run())


def test_document_timeout_cancels_inference_and_releases_lock(client):
    """A timed-out request fails atomically and allows the next document."""
    http, inference = client

    async def blocked(**kwargs):
        """Wait until the service deadline cancels inference."""
        await asyncio.Event().wait()

    app.state.settings.timeout = 0.01
    inference.complete.side_effect = blocked
    response = http.post("/layout-parsing", json=pdf_request(1))
    assert response.status_code == 504
    assert not app.state.lock.locked()
    inference.complete.side_effect = None
    inference.complete.return_value = completion("recovered")
    assert http.post("/layout-parsing", json=pdf_request(1)).status_code == 200


def test_rendering_deadline_prevents_inference(client, monkeypatch):
    """Check the elapsed deadline before inference after synchronous rendering."""
    http, inference = client
    clock = SimpleNamespace(time=Mock(side_effect=[1000.0, 1601.0]))
    monkeypatch.setattr(
        "vision_ocr.app.asyncio",
        SimpleNamespace(
            get_running_loop=lambda: clock,
            timeout_at=lambda _: asyncio.timeout(600),
        ),
    )
    response = http.post("/layout-parsing", json=pdf_request(1))
    assert response.status_code == 504
    inference.complete.assert_not_called()


def test_openrouter_keeps_provider_model_prefix(monkeypatch):
    """Cloud gateway model identifiers retain the provider prefix."""
    monkeypatch.setenv("VISION_OCR_ENDPOINT", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("VISION_OCR_MODEL", "openai/gpt-4o")
    response = completion("transcript")
    response.raw.pop("model")
    inference = SimpleNamespace(
        complete=AsyncMock(return_value=response), close=AsyncMock()
    )
    constructor = Mock(return_value=inference)
    monkeypatch.setattr("vision_ocr.app.InferenceClient", constructor)
    with TestClient(app) as http:
        result = http.post("/layout-parsing", json=pdf_request(1)).json()["result"]
        metadata = http.get("/metadata").json()
    assert constructor.call_args.kwargs["provider"] == "openrouter"
    assert "provenance" not in result
    assert metadata["model"] == "openai/gpt-4o"
