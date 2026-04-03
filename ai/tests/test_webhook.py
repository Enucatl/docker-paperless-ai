"""
Integration tests for the webhook listener service.

Two categories of tests:

1. Listener unit-style tests (fast): POST crafted payloads directly to the
   webhook-listener container and assert Redis queue state.

2. End-to-end Paperless integration tests (slow): configure real Paperless
   Workflows via the API, upload/edit documents, and verify that Paperless
   fires the webhook which lands doc IDs in Redis — proving the full pipeline
   works including {{doc_url}} placeholder rendering.

The webhook-listener and Paperless webserver must be running before pytest
starts — guaranteed by depends_on conditions in docker-compose.test.yml.

Paperless workflow trigger types used in E2E tests:
  2 = DOCUMENT_ADDED    — fires when a new document finishes indexing
  3 = DOCUMENT_UPDATED  — fires when an existing document's metadata is patched
"""

import asyncio
import os
import time
from unittest.mock import patch

import httpx
import pytest

from tests.conftest import (
    PAPERLESS_URL,
    WEBHOOK_URL,
    _make_test_pdf,
    _redis_queue_members,
    _redis_queue_size,
    _redis_stage_members,
    _upload_document,
)
from paperless_ai.search.queue import TaskQueues

# Test webhook secret (must match what tests inject into env)
TEST_WEBHOOK_SECRET = "test-secret-key-12345"


@pytest.fixture
def webhook_with_auth():
    """
    Temporarily enable webhook authentication by patching the global _webhook_secret.
    Yields the secret so tests can use it in headers.
    """
    from paperless_ai.search import webhook as webhook_module

    original_secret = webhook_module._webhook_secret
    webhook_module._webhook_secret = TEST_WEBHOOK_SECRET
    try:
        yield TEST_WEBHOOK_SECRET
    finally:
        webhook_module._webhook_secret = original_secret


@pytest.fixture
def webhook_with_tags():
    """
    Patch the webhook module's tag globals so tests control routing.
    Restores originals on teardown.
    """
    from paperless_ai.search import webhook as webhook_module

    orig_ocr = webhook_module._tag_ocr
    orig_meta = webhook_module._tag_metadata
    orig_embed = webhook_module._tag_embed
    webhook_module._tag_ocr = "ai:run-ocr"
    webhook_module._tag_metadata = "ai:run-metadata"
    webhook_module._tag_embed = "ai:run-embed"
    try:
        yield
    finally:
        webhook_module._tag_ocr = orig_ocr
        webhook_module._tag_metadata = orig_meta
        webhook_module._tag_embed = orig_embed


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_webhook_health(document_queue):
    """GET /health returns 200 with pending counts per stage."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{WEBHOOK_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    pending = body["pending"]
    assert isinstance(pending, dict)
    assert set(pending.keys()) >= {"ocr", "metadata", "embed"}


# ---------------------------------------------------------------------------
# Enqueue via doc_url (primary / recommended Paperless configuration)
# ---------------------------------------------------------------------------


async def test_webhook_enqueues_from_doc_url(document_queue):
    """
    POST with a 'doc_url' field (the {{doc_url}} Jinja2 placeholder that
    Paperless provides) must enqueue the numeric document ID extracted from
    the URL path.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
        )
    assert r.status_code == 202
    assert 42 in _redis_queue_members()


async def test_webhook_enqueues_from_deep_doc_url(document_queue):
    """URL with extra path segments — ID still extracted correctly."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "http://paperless.internal:8000/documents/999/"},
        )
    assert r.status_code == 202
    assert 999 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Enqueue via explicit document_id (fallback for custom webhook bodies)
# ---------------------------------------------------------------------------


async def test_webhook_enqueues_from_document_id_field(document_queue):
    """POST with a plain 'document_id' integer field must enqueue that ID."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"document_id": 77},
        )
    assert r.status_code == 202
    assert 77 in _redis_queue_members()


async def test_webhook_enqueues_from_id_field(document_queue):
    """POST with a plain 'id' integer field (last-resort fallback)."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"id": 55},
        )
    assert r.status_code == 202
    assert 55 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


async def test_webhook_deduplicates_same_id(document_queue):
    """
    Posting the same document URL twice must result in exactly one queue entry.
    Redis SADD is idempotent — this verifies the set-based dedup works end-to-end.
    """
    payload = {"doc_url": "https://paperless.home/documents/100/detail"}
    async with httpx.AsyncClient() as client:
        await client.post(f"{WEBHOOK_URL}/webhook/document", json=payload)
        await client.post(f"{WEBHOOK_URL}/webhook/document", json=payload)

    assert _redis_queue_size() == 1, "Duplicate webhook must not create two queue entries"
    assert 100 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Graceful handling of bad payloads
# ---------------------------------------------------------------------------


async def test_webhook_ignores_payload_without_id(document_queue):
    """
    A payload that carries no recognisable document ID is accepted (202) but
    does not add anything to the queue — Paperless should not be forced to retry.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"event": "document_added", "unrelated": "data"},
        )
    assert r.status_code == 202
    assert _redis_queue_size() == 0


async def test_webhook_rejects_non_json_body(document_queue):
    """A non-JSON body must return 400."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            content=b"not json",
            headers={"Content-Type": "text/plain"},
        )
    assert r.status_code == 400
    assert _redis_queue_size() == 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_webhook_rejects_missing_token(webhook_with_auth, document_queue):
    """When WEBHOOK_SECRET is set, requests without X-Webhook-Token are rejected."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
            # No X-Webhook-Token header
        )
    assert r.status_code == 401
    assert _redis_queue_size() == 0


async def test_webhook_rejects_wrong_token(webhook_with_auth, document_queue):
    """When WEBHOOK_SECRET is set, requests with wrong token are rejected."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
            headers={"X-Webhook-Token": "wrong-secret"},
        )
    assert r.status_code == 401
    assert _redis_queue_size() == 0


async def test_webhook_accepts_correct_token(webhook_with_auth, document_queue):
    """When WEBHOOK_SECRET is set, requests with correct token are accepted."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
            headers={"X-Webhook-Token": TEST_WEBHOOK_SECRET},
        )
    assert r.status_code == 202
    assert 42 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Pending count reflected in health endpoint
# ---------------------------------------------------------------------------


async def test_webhook_health_reflects_pending_count(document_queue):
    """
    After enqueuing two documents, /health must report total pending=2.
    Without document_tags both go to the embed queue (human-edit fallback).
    """
    payloads = [
        {"doc_url": "https://paperless.home/documents/201/detail"},
        {"doc_url": "https://paperless.home/documents/202/detail"},
    ]
    async with httpx.AsyncClient() as client:
        for p in payloads:
            await client.post(f"{WEBHOOK_URL}/webhook/document", json=p)

        r = await client.get(f"{WEBHOOK_URL}/health")

    pending = r.json()["pending"]
    total = sum(pending.values())
    assert total == 2


# ---------------------------------------------------------------------------
# Tag-based routing (Phase B)
# ---------------------------------------------------------------------------


async def test_webhook_routes_ocr_tag_to_ocr_queue(document_queue, webhook_with_tags):
    """ai:run-ocr tag → queue:ocr."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={
                "doc_url": "https://paperless.home/documents/301/detail",
                "document_tags": "ai:run-ocr,invoice",
            },
        )
    assert r.status_code == 202
    assert 301 in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 301 not in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 301 not in _redis_stage_members(TaskQueues.KEY_EMBED)


async def test_webhook_routes_metadata_tag_to_metadata_queue(document_queue, webhook_with_tags):
    """ai:run-metadata tag → queue:metadata."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={
                "doc_url": "https://paperless.home/documents/302/detail",
                "document_tags": "ai:run-metadata",
            },
        )
    assert r.status_code == 202
    assert 302 in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 302 not in _redis_stage_members(TaskQueues.KEY_OCR)


async def test_webhook_routes_embed_tag_to_embed_queue(document_queue, webhook_with_tags):
    """ai:run-embed tag → queue:embed."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={
                "doc_url": "https://paperless.home/documents/303/detail",
                "document_tags": "ai:run-embed",
            },
        )
    assert r.status_code == 202
    assert 303 in _redis_stage_members(TaskQueues.KEY_EMBED)
    assert 303 not in _redis_stage_members(TaskQueues.KEY_OCR)


async def test_webhook_routes_no_ai_tag_to_embed_queue(document_queue, webhook_with_tags):
    """No ai:run-* tag → queue:embed (human edit, keep index in sync)."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={
                "doc_url": "https://paperless.home/documents/304/detail",
                "document_tags": "invoice,personal",
            },
        )
    assert r.status_code == 202
    assert 304 in _redis_stage_members(TaskQueues.KEY_EMBED)


async def test_webhook_routes_no_tags_field_to_embed_queue(document_queue, webhook_with_tags):
    """Missing document_tags key → queue:embed (safe default)."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/305/detail"},
        )
    assert r.status_code == 202
    assert 305 in _redis_stage_members(TaskQueues.KEY_EMBED)


async def test_webhook_ocr_tag_takes_priority_over_embed(document_queue, webhook_with_tags):
    """If both ai:run-ocr and ai:run-embed are present, ocr wins."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={
                "doc_url": "https://paperless.home/documents/306/detail",
                "document_tags": "ai:run-embed,ai:run-ocr",
            },
        )
    assert r.status_code == 202
    assert 306 in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 306 not in _redis_stage_members(TaskQueues.KEY_EMBED)


# ---------------------------------------------------------------------------
# End-to-end: Paperless fires the webhook on document events
# ---------------------------------------------------------------------------

# URL that the Paperless *webserver* container uses to reach the
# webhook-listener (both on the internal Docker network).
# This may differ from WEBHOOK_URL (used by the test runner / ai container)
# if the containers have different DNS views, but in the test compose they
# share the same `internal` network so the hostname resolves identically.
_PAPERLESS_FACING_WEBHOOK_URL = os.environ.get(
    "PAPERLESS_WEBHOOK_URL",
    WEBHOOK_URL,  # same host/port in the test Docker network
)

_TRIGGER_DOCUMENT_ADDED = 2
_TRIGGER_DOCUMENT_UPDATED = 3


@pytest.fixture
def paperless_workflow(paperless_client):
    """
    Factory fixture: create a webhook workflow for each call, delete all on teardown.

    Usage::

        async def test_something(paperless_workflow):
            wf_id = paperless_workflow(_TRIGGER_DOCUMENT_ADDED, "my-test-wf")
            ...
    """
    workflow_ids: list[int] = []

    def _create(trigger_type: int, name: str, filter_has_tags: list | None = None) -> int:
        wf_id = _create_webhook_workflow(paperless_client, trigger_type, name, filter_has_tags)
        workflow_ids.append(wf_id)
        return wf_id

    yield _create

    for wf_id in workflow_ids:
        paperless_client._client.delete(f"/api/workflows/{wf_id}/")


def _create_webhook_workflow(
    client,
    trigger_type: int,
    name: str,
    filter_has_tags: list | None = None,
) -> int:
    """
    Create a Paperless Workflow that POSTs {"doc_url": "{{doc_url}}"} to the
    webhook-listener on the given trigger and return the workflow ID.

    filter_has_tags: list of tag IDs the document must carry for the trigger
    to fire.  Pass the ai-review-pending tag ID here to prevent the workflow
    from re-queuing a document after the AI has removed that tag.
    """
    payload = {
        "name": name,
        "enabled": True,
        "order": 0,
        "triggers": [
            {
                "type": trigger_type,
                "sources": [],  # empty = all sources
                "filter_has_tags": filter_has_tags or [],
            }
        ],
        "actions": [
            {
                "type": 4,  # WEBHOOK
                "webhook": {
                    "url": f"{_PAPERLESS_FACING_WEBHOOK_URL}/webhook/document",
                    "use_params": True,
                    "as_json": True,
                    "params": {"doc_url": "{{doc_url}}"},
                },
            }
        ],
    }
    r = client._client.post("/api/workflows/", json=payload)
    assert r.status_code == 201, (
        f"Workflow creation failed ({trigger_type=}): {r.status_code} — {r.text}"
    )
    return r.json()["id"]


async def _wait_for_doc_in_queue(doc_id: int, timeout: int = 60) -> bool:
    """Poll Redis until doc_id appears in the pending set or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if doc_id in _redis_queue_members():
            return True
        await asyncio.sleep(2)
    return False


async def test_paperless_fires_webhook_on_document_added(
    document_queue, paperless_workflow, uploaded_document
):
    """
    Paperless DOCUMENT_ADDED trigger → webhook → Redis.

    Creates a workflow with trigger type 2 (DOCUMENT_ADDED), uploads a
    document, and verifies the doc ID lands in the Redis queue.  Validates
    the full chain: workflow fires, {{doc_url}} renders, webhook-listener
    extracts the ID and enqueues it.
    """
    paperless_workflow(_TRIGGER_DOCUMENT_ADDED, "test-wf-document-added")
    doc_id = uploaded_document()
    assert await _wait_for_doc_in_queue(doc_id), (
        f"Document {doc_id} never appeared in the Redis queue within 60 s "
        f"(trigger=DOCUMENT_ADDED, webhook={_PAPERLESS_FACING_WEBHOOK_URL})"
    )


async def test_paperless_fires_webhook_on_document_updated(
    paperless_client, document_queue, paperless_workflow, uploaded_document
):
    """
    Paperless DOCUMENT_UPDATED trigger → webhook → Redis.

    Uploads a document with no active webhook workflow (so DOCUMENT_ADDED
    does not pollute the queue), then creates a DOCUMENT_UPDATED workflow
    and PATCHes the document title.  Verifies the doc ID appears in Redis.

    This is the production-relevant trigger: after the AI service patches a
    document's metadata (title, date, correspondent), Paperless fires
    DOCUMENT_UPDATED — confirming that re-processing via the Redis pipeline
    would be triggered unless the AI service removes the pending tag or the
    workflow filters by tag.
    """
    # Upload before creating the workflow so DOCUMENT_ADDED does not fire.
    doc_id = uploaded_document()
    paperless_workflow(_TRIGGER_DOCUMENT_UPDATED, "test-wf-document-updated")

    r = paperless_client._client.patch(
        f"/api/documents/{doc_id}/",
        json={"title": "Updated Title — DOCUMENT_UPDATED webhook test"},
    )
    assert r.status_code == 200, f"PATCH failed: {r.status_code} — {r.text}"
    assert await _wait_for_doc_in_queue(doc_id), (
        f"Document {doc_id} never appeared in the Redis queue within 60 s "
        f"(trigger=DOCUMENT_UPDATED, webhook={_PAPERLESS_FACING_WEBHOOK_URL}). "
        "Check that trigger type 3 is DOCUMENT_UPDATED in this Paperless version."
    )


async def test_document_updated_deduplicates_repeated_edits(
    paperless_client, document_queue, paperless_workflow, uploaded_document
):
    """
    Multiple rapid edits to the same document produce exactly one queue entry.

    The webhook fires on each DOCUMENT_UPDATED event, but Redis SADD is
    idempotent — the pending set must contain the doc ID exactly once even
    after three consecutive PATCHes.
    """
    doc_id = uploaded_document()
    paperless_workflow(_TRIGGER_DOCUMENT_UPDATED, "test-wf-dedup-updates")

    for i in range(3):
        r = paperless_client._client.patch(
            f"/api/documents/{doc_id}/", json={"title": f"Edit {i + 1}"}
        )
        assert r.status_code == 200

    assert await _wait_for_doc_in_queue(doc_id), (
        f"Document {doc_id} not queued after repeated edits"
    )
    # Let any additional webhook calls settle, then check the count is still 1.
    await asyncio.sleep(3)
    assert _redis_queue_size() == 1, (
        f"Expected exactly 1 queue entry after 3 edits, got {_redis_queue_size()}"
    )


async def test_webhook_loop_broken_by_tag_removal(
    paperless_client, document_queue, paperless_workflow, uploaded_document
):
    """
    After run_batch processes a document it removes the ai-review-pending tag
    atomically in the same PATCH as the metadata.  A DOCUMENT_UPDATED workflow
    filtered by that tag therefore does not re-queue the document.

    Flow:
    1. Upload document and manually add the ai-review-pending tag.
    2. Create a DOCUMENT_UPDATED workflow that only fires while the tag is present.
    3. Enqueue the doc and run_batch — AI patches metadata + removes tag.
    4. Edit the document title (simulating any further change).
    5. Assert the queue remains empty — the tag filter blocked re-queuing.
    """
    from paperless_ai.agents.smart_graph_agent import SmartDocumentAgent, _select_extraction_strategy
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        ocr_model="gemini/gemini-2.5-flash",
    )
    agent = SmartDocumentAgent(config, extraction_strategy=_select_extraction_strategy(config))

    custom_field_id = paperless_client.get_or_create_custom_field("ai_processed", data_type="date")
    ai_result_field_id = paperless_client.get_or_create_custom_field("ai_result", data_type="longtext")

    tag_id = paperless_client.get_tag_id(config.tag_pending)

    # Upload before creating the workflow so DOCUMENT_ADDED does not fire.
    doc_id = uploaded_document()

    # Add the pending tag — this is what the DOCUMENT_ADDED assignment action does.
    r = paperless_client._client.patch(f"/api/documents/{doc_id}/", json={"tags": [tag_id]})
    assert r.status_code == 200, f"Failed to add pending tag: {r.text}"

    paperless_workflow(_TRIGGER_DOCUMENT_UPDATED, "test-wf-loop-guard", filter_has_tags=[tag_id])

    # Enqueue and process — run_batch removes the tag in the same PATCH.
    await document_queue.enqueue(doc_id)
    success, failure = await run_batch(
        paperless_client, agent, config,
        custom_field_id, ai_result_field_id,
        document_queue,
    )
    assert success == 1 and failure == 0, f"run_batch: {success=} {failure=}"

    doc = paperless_client.get_document(doc_id)
    assert tag_id not in doc.get("tags", []), (
        "ai-review-pending tag was not removed by run_batch"
    )

    # Wait for any in-flight webhook from the processing PATCH to settle.
    await asyncio.sleep(5)
    assert _redis_queue_size() == 0, (
        f"Queue not empty after run_batch — DOCUMENT_UPDATED may have re-queued "
        f"doc {doc_id} despite tag removal"
    )

    # Simulate a further edit (e.g. user renames the document manually).
    r = paperless_client._client.patch(
        f"/api/documents/{doc_id}/", json={"title": "Manually renamed"}
    )
    assert r.status_code == 200

    # Allow time for the webhook to fire (it should not — tag is gone).
    await asyncio.sleep(5)
    assert _redis_queue_size() == 0, (
        "Queue grew after post-processing edit — tag filter did not block re-queue"
    )
