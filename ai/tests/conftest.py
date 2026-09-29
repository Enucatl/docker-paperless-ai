"""
Pytest fixtures for the paperless-ai E2E test suite.

Session-scoped fixtures handle the slow operations (waiting for Paperless,
fetching an API token) once per run.  Function-scoped fixtures isolate
document state between tests and reset mocks cleanly.

Infrastructure available in the test environment (docker-compose.test.yml):
  - Paperless-ngx (webserver)    http://webserver:8000
  - Redis (broker)               redis://broker:6379/1  (DB 1, AI queue)
  - Webhook listener             http://webhook-listener:8001
"""

import asyncio
import io
import json
import os
import time
import uuid
from unittest.mock import patch
from shared_inference import CompletionResult, Usage

import niquests
import pytest
import redis as _redis_sync


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis to be running"
    )
    config.addinivalue_line(
        "markers",
        "requires_webhook_listener: mark test as requiring webhook-listener to be running",
    )


PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
REDIS_URL = os.environ.get("REDIS_URL", "redis://broker:6379/1")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://webhook-listener:8001")
TEST_USER = os.environ.get("TEST_PAPERLESS_USER", "admin")
TEST_PASS = os.environ.get("TEST_PAPERLESS_PASS", "admin")

# OCR and metadata task queues
_TASK_QUEUE_KEYS = [
    "paperless-ai:queue:ocr",
    "paperless-ai:queue:metadata",
]
_ALL_QUEUE_KEYS = [
    *_TASK_QUEUE_KEYS,
    "paperless-ai:queue:failed",
    *[f"paperless-ai:queue:delayed:{key}" for key in _TASK_QUEUE_KEYS],
]


# ---------------------------------------------------------------------------
# Sync Redis helpers (used in sync fixtures and teardown)
# ---------------------------------------------------------------------------


def _redis_client() -> _redis_sync.Redis:
    return _redis_sync.from_url(REDIS_URL, decode_responses=False)


def _clear_redis_queue() -> None:
    """Clear all stage queues."""
    r = _redis_client()
    for key in _ALL_QUEUE_KEYS:
        r.delete(key)
    r.close()


def _redis_queue_size() -> int:
    """Return total pending count across all queues."""
    r = _redis_client()
    total = sum(int(r.scard(key)) for key in _TASK_QUEUE_KEYS)
    r.close()
    return total


def _redis_queue_members() -> set[int]:
    """Return all pending doc IDs across all queues."""
    r = _redis_client()
    members: set[int] = set()
    for key in _TASK_QUEUE_KEYS:
        members.update(int(m) for m in r.smembers(key))
    r.close()
    return members


def _redis_stage_members(stage: str) -> set[int]:
    """Return pending doc IDs for a specific stage queue key."""
    r = _redis_client()
    members = r.smembers(stage)
    r.close()
    return {int(m) for m in members}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_paperless(url: str, timeout: int = 5) -> None:
    """Block until the Paperless API responds to /api/ (or timeout expires)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = niquests.get(f"{url}/api/", timeout=2, allow_redirects=True)
            if r.status_code < 500:
                return
        except niquests.RequestException, niquests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Paperless not ready at {url} after {timeout}s")


def _fetch_token(url: str, user: str, password: str, retries: int = 20) -> str:
    """Obtain an API token via Basic Auth. Retries because user creation is async."""
    import logging

    log = logging.getLogger(__name__)
    for attempt in range(retries):
        try:
            r = niquests.post(
                f"{url}/api/token/",
                json={"username": user, "password": password},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["token"]
            log.warning(
                "_fetch_token: attempt %d/%d status=%d",
                attempt + 1,
                retries,
                r.status_code,
            )
        except (niquests.ConnectionError, niquests.Timeout) as e:
            log.warning(
                "_fetch_token: attempt %d/%d connection error: %s",
                attempt + 1,
                retries,
                e,
            )
        time.sleep(3)
    raise RuntimeError(f"Could not obtain Paperless API token after {retries} attempts")


# ---------------------------------------------------------------------------
# Session-scoped: shared across the whole test run
# ---------------------------------------------------------------------------


def _redis_available() -> bool:
    """Check if Redis is available."""
    try:
        r = _redis_client()
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def _webhook_listener_available() -> bool:
    """Check if webhook-listener is available."""
    try:
        r = niquests.get(f"{WEBHOOK_URL}/health", timeout=2.0)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="session")
def redis_available():
    """Fixture that returns whether Redis is available."""
    return _redis_available()


def pytest_runtest_setup(item):
    """Skip tests that require infrastructure if not available."""
    if "requires_redis" in item.keywords:
        if not _redis_available():
            pytest.skip("Redis is not available")
    if "requires_webhook_listener" in item.keywords:
        if not _webhook_listener_available():
            pytest.skip("webhook-listener is not available")


@pytest.fixture(scope="session")
def paperless_token() -> str:
    try:
        _wait_for_paperless(PAPERLESS_URL)
        return _fetch_token(PAPERLESS_URL, TEST_USER, TEST_PASS)
    except RuntimeError as e:
        pytest.skip(f"Paperless infrastructure not available: {e}")


@pytest.fixture
async def paperless_client(paperless_token: str):
    from paperless_common.paperless import PaperlessClient

    async with PaperlessClient(PAPERLESS_URL, paperless_token) as client:
        yield client


# ---------------------------------------------------------------------------
# Function-scoped: reset inference mock for every test
# ---------------------------------------------------------------------------

# Deterministic LLM responses used by all tests.
_OCR_TEXT = "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024\nTotal: $100.00"
_METADATA_JSON = json.dumps(
    {
        "title": "Test Invoice",
        "document_date": "2024-01-15",
        "correspondent": "Acme Corp",
        "summary": "Invoice from Acme Corp dated 2024-01-15 for $100.00.",
    }
)


def _make_fake_completion():
    async def fake_completion(**kwargs):
        # Document OCR is mocked separately at the document-service interface.
        content = _METADATA_JSON
        return CompletionResult(
            content=content,
            message={"role": "assistant", "content": content},
            tool_calls=[],
            reasoning=None,
            usage=Usage(),
            request_id=None,
            raw={},
        )

    return fake_completion


@pytest.fixture(autouse=True)
def mock_litellm():
    """
    Intercept shared inference completion calls with deterministic responses.
    """
    fake = _make_fake_completion()
    with (
        patch("paperless_ai.agents.smart_graph_agent.complete", side_effect=fake),
        patch("paperless_ai.inference.complete", side_effect=fake),
    ):
        yield


@pytest.fixture
def mock_document_service():
    """Return deterministic complete-document OCR for pipeline integration tests."""
    from unittest.mock import AsyncMock

    output = {
        "schema_version": 1,
        "pipeline": "test-document-service",
        "model": "test-ocr",
        "layout_model": "test-layout",
        "page_count": 1,
        "pages": [{"page_index": 0, "markdown": {"text": _OCR_TEXT}}],
    }
    with (
        patch(
            "paperless_ai.agents.paddle_ocr.run_paddle_ocr",
            new=AsyncMock(return_value=(_OCR_TEXT, output, 1, 0.01)),
        ),
        patch(
            "paperless_ai.core.runner._check_server_reachable",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "paperless_ai.agents.paddle_ocr.fetch_ocr_metadata",
            new=AsyncMock(
                return_value={
                    "pipeline": "test-document-service",
                    "model": "test-ocr",
                    "layout_model": "test-layout",
                }
            ),
        ),
    ):
        yield


# ---------------------------------------------------------------------------
# Function-scoped: Redis queues — cleared before and after each test
# ---------------------------------------------------------------------------


@pytest.fixture
async def task_queues():
    """
    A fresh TaskQueues backed by the real test Redis (DB 1).

    Clears both stage queues before and after each test.
    """
    from paperless_common.queue import TaskQueues

    if not _redis_available():
        pytest.skip("Redis is not available")

    _clear_redis_queue()
    q = TaskQueues(REDIS_URL)
    yield q
    await q.close()
    _clear_redis_queue()


# ---------------------------------------------------------------------------
# Function-scoped: upload a fresh document for each test that needs one
# ---------------------------------------------------------------------------


def _make_test_pdf(seed: str | None = None) -> bytes:
    """Generate a tiny PDF with native digital text using PyMuPDF."""
    import fitz

    marker = seed or uuid.uuid4().hex
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 700),
        (
            "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024\n"
            f"Total: $100.00\nTest Marker: {marker}"
        ),
        fontname="helv",
        fontsize=12,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


async def _upload_document(client, pdf_bytes: bytes) -> int:
    """Upload a PDF to Paperless and wait for it to be indexed. Returns doc_id."""
    filename = f"dummy_invoice_{uuid.uuid4().hex}.pdf"
    r = await client._client.post(
        "/api/documents/post_document/",
        files={"document": (filename, pdf_bytes, "application/pdf")},
        timeout=10,
    )
    r.raise_for_status()

    # Paperless returns the task UUID as a bare quoted string, e.g. "abc-123"
    raw = r.text.strip()
    task_uuid = raw.strip('"')

    # Tiny native-text PDFs finish in seconds; surface consumer failures promptly.
    deadline = time.monotonic() + 10
    task = None
    while time.monotonic() < deadline:
        tasks_r = await client._client.get(
            "/api/tasks/", params={"task_id": task_uuid}, timeout=2
        )
        tasks_r.raise_for_status()
        tasks = tasks_r.json()
        if isinstance(tasks, dict):
            tasks = tasks["results"]
        task = next((task for task in tasks if task["task_id"] == task_uuid), None)
        if task:
            status = task["status"].lower()
            if status == "success":
                # v10 uses structured results; v9 exposes related_document.
                doc_id = (task.get("result_data") or {}).get("document_id") or task.get(
                    "related_document"
                )
                if doc_id is not None:
                    return int(doc_id)
                raise RuntimeError(
                    f"Paperless task completed without a document: {task}"
                )
            if status in {"failure", "revoked"}:
                raise RuntimeError(f"Paperless task {task_uuid} failed: {task}")
        await asyncio.sleep(0.1)

    raise RuntimeError(
        f"Document not indexed after 10 s (task={task_uuid}, last_response={task})"
    )
