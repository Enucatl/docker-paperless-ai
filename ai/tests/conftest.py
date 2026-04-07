"""
Pytest fixtures for the paperless-ai E2E test suite.

Session-scoped fixtures handle the slow operations (waiting for Paperless,
fetching an API token) once per run.  Function-scoped fixtures isolate
document state between tests and reset mocks cleanly.

Infrastructure available in the test environment (docker-compose.test.yml):
  - Paperless-ngx (webserver)    http://webserver:8000
  - Redis (broker)               redis://broker:6379/1  (DB 1, AI queue)
  - Qdrant (vector DB)           http://qdrant:6333
  - Webhook listener             http://webhook-listener:8001
  - Infinity embedder            NOT available — use mock_embedder fixture instead
"""

import io
import json
import os
import time
from unittest.mock import MagicMock, patch

import niquests
import pytest
import redis as _redis_sync


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "requires_redis: mark test as requiring Redis to be running"
    )
    config.addinivalue_line(
        "markers", "requires_webhook_listener: mark test as requiring webhook-listener to be running"
    )

PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
REDIS_URL = os.environ.get("REDIS_URL", "redis://broker:6379/1")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://webhook-listener:8001")
TEST_USER = os.environ.get("TEST_PAPERLESS_USER", "admin")
TEST_PASS = os.environ.get("TEST_PAPERLESS_PASS", "admin")

_QUEUE_KEY = "paperless-ai:pending"

# Phase B: three-stage task queues
_TASK_QUEUE_KEYS = [
    "paperless-ai:queue:ocr",
    "paperless-ai:queue:metadata",
    "paperless-ai:queue:embed",
]
# All queue keys — old + new — used by aggregate helpers
_ALL_QUEUE_KEYS = [_QUEUE_KEY] + _TASK_QUEUE_KEYS


# ---------------------------------------------------------------------------
# Sync Redis helpers (used in sync fixtures and teardown)
# ---------------------------------------------------------------------------


def _redis_client() -> _redis_sync.Redis:
    return _redis_sync.from_url(REDIS_URL, decode_responses=False)


def _clear_redis_queue() -> None:
    """Clear all queues (old single queue + three task queues)."""
    r = _redis_client()
    for key in _ALL_QUEUE_KEYS:
        r.delete(key)
    r.close()


def _redis_enqueue(doc_id: int) -> None:
    r = _redis_client()
    r.sadd(_QUEUE_KEY, doc_id)
    r.close()


def _redis_queue_size() -> int:
    """Return total pending count across all queues."""
    r = _redis_client()
    total = sum(int(r.scard(key)) for key in _ALL_QUEUE_KEYS)
    r.close()
    return total


def _redis_queue_members() -> set[int]:
    """Return all pending doc IDs across all queues."""
    r = _redis_client()
    members: set[int] = set()
    for key in _ALL_QUEUE_KEYS:
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
        except (niquests.RequestException, niquests.ConnectionError):
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
            log.warning("_fetch_token: attempt %d/%d status=%d", attempt + 1, retries, r.status_code)
        except (niquests.ConnectionError, niquests.Timeout) as e:
            log.warning("_fetch_token: attempt %d/%d connection error: %s", attempt + 1, retries, e)
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


def _qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        r = niquests.get(f"{QDRANT_URL}/health", timeout=2.0)
        return r.status_code < 500
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
    from paperless_ai.core.paperless import PaperlessClient

    async with PaperlessClient(PAPERLESS_URL, paperless_token) as client:
        yield client


# ---------------------------------------------------------------------------
# Function-scoped: reset LiteLLM mock for every test
# ---------------------------------------------------------------------------

# Deterministic LLM responses used by all tests.
_OCR_TEXT = (
    "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024\nTotal: $100.00"
)
_METADATA_JSON = json.dumps({
    "title": "Test Invoice",
    "document_date": "2024-01-15",
    "correspondent": "Acme Corp",
    "summary": "Invoice from Acme Corp dated 2024-01-15 for $100.00.",
})


def _make_fake_acompletion():
    async def fake_acompletion(**kwargs):
        resp = MagicMock()
        msg = MagicMock()
        # Metadata calls set response_format; OCR calls do not.
        if kwargs.get("response_format") is not None:
            msg.content = _METADATA_JSON
        else:
            msg.content = _OCR_TEXT
        resp.choices = [MagicMock(message=msg)]
        return resp

    return fake_acompletion


@pytest.fixture(autouse=True)
def mock_litellm():
    """
    Intercept every litellm.acompletion call with a deterministic response.

    Patches at the top-level litellm module so that any code doing
    `import litellm; await litellm.acompletion(...)` is intercepted.
    """
    fake = _make_fake_acompletion()
    with patch("litellm.acompletion", side_effect=fake):
        yield


# ---------------------------------------------------------------------------
# Function-scoped: Redis queue — cleared before and after each test
# ---------------------------------------------------------------------------


@pytest.fixture
async def document_queue():
    """
    A fresh DocumentQueue backed by the real test Redis (DB 1).

    Clears the queue before the test starts so each test runs against a
    known-empty queue.  Cleans up afterwards regardless of test outcome.
    """
    from paperless_ai.search.queue import DocumentQueue

    if not _redis_available():
        pytest.skip("Redis is not available")

    _clear_redis_queue()
    q = DocumentQueue(REDIS_URL)
    yield q
    await q.close()
    _clear_redis_queue()


@pytest.fixture
async def task_queues():
    """
    A fresh TaskQueues backed by the real test Redis (DB 1).

    Clears all three stage queues before and after each test.
    """
    from paperless_ai.search.queue import TaskQueues

    if not _redis_available():
        pytest.skip("Redis is not available")

    _clear_redis_queue()
    q = TaskQueues(REDIS_URL)
    yield q
    await q.close()
    _clear_redis_queue()


# ---------------------------------------------------------------------------
# Function-scoped: mock embedder — deterministic vectors, no network calls
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder(monkeypatch):
    """
    A fake InfinityEmbedder that returns deterministic 1024-d dense vectors
    and sparse BM25 weights without making any network calls.

    Pass this directly to run_batch() when testing the embedding pipeline.
    Infinity is not available in the test environment.

    Also patches _check_server_reachable so that run_embed_batch's preflight
    check doesn't bail early — that check is a production guard and is
    meaningless when a mock embedder is in use.
    """
    from paperless_ai.core import runner as _runner

    async def _always_reachable(url: str) -> bool:  # noqa: ARG001
        return True

    monkeypatch.setattr(_runner, "_check_server_reachable", _always_reachable)
    from paperless_ai.search.embedder import EmbeddingResult

    class _MockEmbedder:
        async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
            return [
                EmbeddingResult(
                    dense=[0.01 * (i % 100)] * 1024,
                    sparse_indices=[1, 42, 512],
                    sparse_values=[0.7, 0.3, 0.1],
                )
                for i, _ in enumerate(texts)
            ]

        async def aclose(self) -> None:
            """No-op stub for compatibility with async context manager."""
            pass

    return _MockEmbedder()


# ---------------------------------------------------------------------------
# Function-scoped: Qdrant store — real Qdrant, cleaned up per test
# ---------------------------------------------------------------------------


@pytest.fixture
async def uploaded_document(paperless_client):
    """
    Factory fixture: upload a fresh document for each call, delete all on teardown.

    Unlike dummy_document, this fixture does NOT enqueue the document in Redis —
    useful for tests that manage the queue themselves (e.g. webhook E2E tests).

    Usage::

        async def test_something(uploaded_document):
            doc_id = await uploaded_document()
            ...
    """
    doc_ids: list[int] = []

    async def _upload() -> int:
        doc_id = await _upload_document(paperless_client, _make_test_pdf())
        doc_ids.append(doc_id)
        return doc_id

    yield _upload

    for doc_id in doc_ids:
        try:
            await paperless_client._client.delete(f"/api/documents/{doc_id}/")
        except Exception:
            pass


@pytest.fixture
async def qdrant_store():
    """
    A QdrantDocumentStore connected to the test Qdrant instance.

    Creates the collection if it does not exist.  The collection persists
    within the test run (anonymous volume); `docker compose down -v` wipes it.
    """
    from paperless_ai.search.qdrant_store import QdrantDocumentStore

    if not _qdrant_available():
        pytest.skip("Qdrant is not available")

    store = QdrantDocumentStore(url=QDRANT_URL)
    await store.ensure_collection()
    yield store


# ---------------------------------------------------------------------------
# Function-scoped: upload a fresh document for each test that needs one
# ---------------------------------------------------------------------------


def _make_test_pdf() -> bytes:
    """Generate a tiny PDF with native digital text using PyMuPDF."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        (72, 700),
        "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024\nTotal: $100.00",
        fontname="helv",
        fontsize=12,
    )
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


async def _upload_document(client, pdf_bytes: bytes) -> int:
    """Upload a PDF to Paperless and wait for it to be indexed. Returns doc_id."""
    r = await client._client.post(
        "/api/documents/post_document/",
        files={"document": ("dummy_invoice.pdf", pdf_bytes, "application/pdf")},
        timeout=30,
    )
    r.raise_for_status()

    # Paperless returns the task UUID as a bare quoted string, e.g. "abc-123"
    raw = r.text.strip()
    task_uuid = raw.strip('"')

    # Poll the task until it completes (up to 120 s)
    doc_id = None
    for _ in range(60):
        time.sleep(2)
        tasks_r = await client._client.get("/api/tasks/", params={"task_id": task_uuid})
        tasks_r.raise_for_status()
        tasks = tasks_r.json()
        if tasks:
            task = tasks[0]
            status = task.get("status", "")
            if status == "SUCCESS":
                # "related_document" is an integer id when available;
                # "result" may be a string like "Success. New document id 4 created"
                doc_id = task.get("related_document")
                if doc_id is None:
                    import re
                    m = re.search(r"\b(\d+)\b", str(task.get("result", "")))
                    if m:
                        doc_id = int(m.group(1))
                break
            if status == "FAILURE":
                raise RuntimeError(f"Paperless task {task_uuid} failed: {task}")

    if doc_id is None:
        raise RuntimeError(
            f"Document not indexed after 120 s (task={task_uuid})"
        )
    return int(doc_id)  # may already be int if from related_document


@pytest.fixture
async def dummy_document(paperless_client, document_queue):
    """
    Upload a test PDF to Paperless, enqueue its ID in the Redis queue, yield
    the document ID, and clean up (delete the document) after the test.

    Takes `document_queue` as a dependency so the queue is guaranteed to be
    cleared before the document ID is enqueued.
    """
    pdf_bytes = _make_test_pdf()
    doc_id = await _upload_document(paperless_client, pdf_bytes)

    # Enqueue the document ID so run_batch() will pick it up
    await document_queue.enqueue(doc_id)

    yield doc_id

    # Cleanup: delete the document so each test starts from a clean state
    try:
        await paperless_client._client.delete(f"/api/documents/{doc_id}/")
    except Exception:
        pass
