"""
Pytest fixtures for the paperless-ai E2E test suite.

Session-scoped fixtures handle the slow operations (waiting for Paperless,
fetching an API token) once per run. Function-scoped fixtures isolate
document state between tests and reset mocks cleanly.
"""

import io
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Ensure /app is importable regardless of the pytest working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
TEST_USER = os.environ.get("TEST_PAPERLESS_USER", "admin")
TEST_PASS = os.environ.get("TEST_PAPERLESS_PASS", "admin")


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
                # `result` is the doc_id (int or stringified int)
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
def dummy_document(paperless_client):
    """
    Upload a test PDF to Paperless, tag it with ai-review-pending, yield the
    document ID, and clean up (delete the document) after the test.
    """
    from core.config import AgentConfig

    pdf_bytes = _make_test_pdf()
    doc_id = _upload_document(paperless_client, pdf_bytes)

    # Tag with ai-review-pending
    pending_id = paperless_client.get_tag_id("ai-review-pending", create=True)
    doc_r = paperless_client._client.get(f"/api/documents/{doc_id}/")
    doc_r.raise_for_status()
    doc_data = doc_r.json()
    current_tags = doc_data.get("tags", [])
    if pending_id not in current_tags:
        paperless_client._client.patch(
            f"/api/documents/{doc_id}/",
            json={"tags": current_tags + [pending_id]},
        ).raise_for_status()

    yield doc_id

    # Cleanup: delete the document so each test starts from a clean state
    try:
        paperless_client._client.delete(f"/api/documents/{doc_id}/")
    except Exception:
        pass
