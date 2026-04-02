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
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import redis as _redis_sync

# Ensure /app is importable regardless of the pytest working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
REDIS_URL = os.environ.get("REDIS_URL", "redis://broker:6379/1")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "http://webhook-listener:8001")
TEST_USER = os.environ.get("TEST_PAPERLESS_USER", "admin")
TEST_PASS = os.environ.get("TEST_PAPERLESS_PASS", "admin")

_QUEUE_KEY = "paperless-ai:pending"


# ---------------------------------------------------------------------------
# Sync Redis helpers (used in sync fixtures and teardown)
# ---------------------------------------------------------------------------


def _redis_client() -> _redis_sync.Redis:
    return _redis_sync.from_url(REDIS_URL, decode_responses=False)


def _clear_redis_queue() -> None:
    r = _redis_client()
    r.delete(_QUEUE_KEY)
    r.close()


def _redis_enqueue(doc_id: int) -> None:
    r = _redis_client()
    r.sadd(_QUEUE_KEY, doc_id)
    r.close()


def _redis_queue_size() -> int:
    r = _redis_client()
    result = r.scard(_QUEUE_KEY)
    r.close()
    return int(result)


def _redis_queue_members() -> set[int]:
    r = _redis_client()
    members = r.smembers(_QUEUE_KEY)
    r.close()
    return {int(m) for m in members}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_paperless(url: str, timeout: int = 180) -> None:
    """Block until the Paperless API responds to /api/ (or timeout expires)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{url}/api/", timeout=5, follow_redirects=True)
            if r.status_code < 500:
                return
        except httpx.RequestError:
            pass
        time.sleep(3)
    raise RuntimeError(f"Paperless not ready at {url} after {timeout}s")


def _fetch_token(url: str, user: str, password: str, retries: int = 20) -> str:
    """Obtain an API token via Basic Auth. Retries because user creation is async."""
    for attempt in range(retries):
        try:
            r = httpx.post(
                f"{url}/api/token/",
                json={"username": user, "password": password},
                timeout=15,
            )
            if r.status_code == 200:
                return r.json()["token"]
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"Could not obtain Paperless API token after {retries} attempts")


# ---------------------------------------------------------------------------
# Session-scoped: shared across the whole test run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def paperless_token() -> str:
    _wait_for_paperless(PAPERLESS_URL)
    return _fetch_token(PAPERLESS_URL, TEST_USER, TEST_PASS)


@pytest.fixture(scope="session")
def paperless_client(paperless_token: str):
    from core.paperless import PaperlessClient

    client = PaperlessClient(PAPERLESS_URL, paperless_token)
    yield client
    client.close()


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
    from search.queue import DocumentQueue

    _clear_redis_queue()
    q = DocumentQueue(REDIS_URL)
    yield q
    await q.close()
    _clear_redis_queue()


# ---------------------------------------------------------------------------
# Function-scoped: mock embedder — deterministic vectors, no network calls
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder():
    """
    A fake InfinityEmbedder that returns deterministic 1024-d dense vectors
    and sparse BM25 weights without making any network calls.

    Pass this directly to run_batch() when testing the embedding pipeline.
    Infinity is not available in the test environment.
    """
    from search.embedder import EmbeddingResult

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

    return _MockEmbedder()


# ---------------------------------------------------------------------------
# Function-scoped: Qdrant store — real Qdrant, cleaned up per test
# ---------------------------------------------------------------------------


@pytest.fixture
async def qdrant_store():
    """
    A QdrantDocumentStore connected to the test Qdrant instance.

    Creates the collection if it does not exist.  The collection persists
    within the test run (anonymous volume); `docker compose down -v` wipes it.
    """
    from search.qdrant_store import QdrantDocumentStore

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


def _upload_document(client, pdf_bytes: bytes) -> int:
    """Upload a PDF to Paperless and wait for it to be indexed. Returns doc_id."""
    r = client._client.post(
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
        tasks_r = client._client.get("/api/tasks/", params={"task_id": task_uuid})
        tasks_r.raise_for_status()
        tasks = tasks_r.json()
        if tasks:
            task = tasks[0]
            status = task.get("status", "")
            if status == "SUCCESS":
                doc_id = task.get("result") or task.get("related_document")
                break
            if status == "FAILURE":
                raise RuntimeError(f"Paperless task {task_uuid} failed: {task}")

    if doc_id is None:
        raise RuntimeError(
            f"Document not indexed after 120 s (task={task_uuid})"
        )
    return int(doc_id)


@pytest.fixture
async def dummy_document(paperless_client, document_queue):
    """
    Upload a test PDF to Paperless, enqueue its ID in the Redis queue, yield
    the document ID, and clean up (delete the document) after the test.

    Takes `document_queue` as a dependency so the queue is guaranteed to be
    cleared before the document ID is enqueued.
    """
    pdf_bytes = _make_test_pdf()
    doc_id = _upload_document(paperless_client, pdf_bytes)

    # Enqueue the document ID so run_batch() will pick it up
    await document_queue.enqueue(doc_id)

    yield doc_id

    # Cleanup: delete the document so each test starts from a clean state
    try:
        paperless_client._client.delete(f"/api/documents/{doc_id}/")
    except Exception:
        pass
