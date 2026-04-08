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

import niquests
import pytest

from tests.conftest import (
    PAPERLESS_URL,
    WEBHOOK_URL,
    _make_test_pdf,
    _redis_client,
    _redis_queue_members,
    _redis_queue_size,
    _redis_stage_members,
    _upload_document,
)
from paperless_ai.search.queue import TaskQueues

# Test webhook secret — must match WEBHOOK_SECRET in docker-compose.test.yml.
TEST_WEBHOOK_SECRET = "test-secret-key-12345"
_AUTH_HEADERS = {"X-Webhook-Token": TEST_WEBHOOK_SECRET}


@pytest.fixture
async def webhook_session():
    """niquests session pre-configured with the webhook auth token."""
    async with niquests.AsyncSession() as s:
        s.headers.update(_AUTH_HEADERS)
        yield s


@pytest.fixture
def webhook_with_tags():
    """
    Patch the webhook module's tag globals so tests control routing.
    Restores originals on teardown.
    """
    from paperless_listener import app as webhook_module

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


async def test_webhook_health(task_queues):
    """GET /health returns 200 with pending counts per stage."""
    async with niquests.AsyncSession() as client:
        r = await client.get(f"{WEBHOOK_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    pending = body["pending"]
    assert isinstance(pending, dict)
    assert set(pending.keys()) >= {"ocr", "metadata", "embed", "refresh"}


# ---------------------------------------------------------------------------
# Enqueue via doc_url (primary / recommended Paperless configuration)
# ---------------------------------------------------------------------------


async def test_webhook_enqueues_from_doc_url(webhook_session, task_queues):
    """
    POST with a 'doc_url' field (the {{doc_url}} Jinja2 placeholder that
    Paperless provides) must enqueue the numeric document ID extracted from
    the URL path.
    """
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"doc_url": "https://paperless.home/documents/42/detail"},
    )
    assert r.status_code == 202
    assert 42 in _redis_queue_members()


async def test_webhook_enqueues_from_deep_doc_url(webhook_session, task_queues):
    """URL with extra path segments — ID still extracted correctly."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"doc_url": "http://paperless.internal:8000/documents/999/"},
    )
    assert r.status_code == 202
    assert 999 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Enqueue via explicit document_id (fallback for custom webhook bodies)
# ---------------------------------------------------------------------------


async def test_webhook_enqueues_from_document_id_field(webhook_session, task_queues):
    """POST with a plain 'document_id' integer field must enqueue that ID."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"document_id": 77},
    )
    assert r.status_code == 202
    assert 77 in _redis_queue_members()


async def test_webhook_enqueues_from_id_field(webhook_session, task_queues):
    """POST with a plain 'id' integer field (last-resort fallback)."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"id": 55},
    )
    assert r.status_code == 202
    assert 55 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


async def test_webhook_deduplicates_same_id(webhook_session, task_queues):
    """
    Posting the same document URL twice must result in exactly one queue entry.
    Redis SADD is idempotent — this verifies the set-based dedup works end-to-end.
    """
    payload = {"doc_url": "https://paperless.home/documents/100/detail"}
    await webhook_session.post(f"{WEBHOOK_URL}/webhook/document", json=payload)
    await webhook_session.post(f"{WEBHOOK_URL}/webhook/document", json=payload)

    assert _redis_queue_size() == 1, "Duplicate webhook must not create two queue entries"
    assert 100 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Graceful handling of bad payloads
# ---------------------------------------------------------------------------


async def test_webhook_ignores_payload_without_id(webhook_session, task_queues):
    """
    A payload that carries no recognisable document ID is accepted (202) but
    does not add anything to the queue — Paperless should not be forced to retry.
    """
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"event": "document_added", "unrelated": "data"},
    )
    assert r.status_code == 202
    assert _redis_queue_size() == 0


async def test_webhook_rejects_non_json_body(webhook_session, task_queues):
    """A non-JSON body must return 400."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        data=b"not json",
        headers={"Content-Type": "text/plain"},
    )
    assert r.status_code == 400
    assert _redis_queue_size() == 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def test_webhook_rejects_missing_token(task_queues):
    """When WEBHOOK_SECRET is set, requests without X-Webhook-Token are rejected."""
    async with niquests.AsyncSession() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
        )
    assert r.status_code == 401
    assert _redis_queue_size() == 0


async def test_webhook_rejects_wrong_token(task_queues):
    """When WEBHOOK_SECRET is set, requests with wrong token are rejected."""
    async with niquests.AsyncSession() as client:
        r = await client.post(
            f"{WEBHOOK_URL}/webhook/document",
            json={"doc_url": "https://paperless.home/documents/42/detail"},
            headers={"X-Webhook-Token": "wrong-secret"},
        )
    assert r.status_code == 401
    assert _redis_queue_size() == 0


async def test_webhook_accepts_correct_token(webhook_session, task_queues):
    """When WEBHOOK_SECRET is set, requests with correct token are accepted."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"doc_url": "https://paperless.home/documents/42/detail"},
    )
    assert r.status_code == 202
    assert 42 in _redis_queue_members()


# ---------------------------------------------------------------------------
# Pending count reflected in health endpoint
# ---------------------------------------------------------------------------


async def test_webhook_health_reflects_pending_count(webhook_session, task_queues):
    """
    Untagged webhook payloads enqueue refresh work.
    """
    payloads = [
        {"doc_url": "https://paperless.home/documents/201/detail"},
        {"doc_url": "https://paperless.home/documents/202/detail"},
    ]
    for p in payloads:
        await webhook_session.post(f"{WEBHOOK_URL}/webhook/document", json=p)

    async with niquests.AsyncSession() as client:
        r = await client.get(f"{WEBHOOK_URL}/health")

    pending = r.json()["pending"]
    total = sum(pending.values())
    assert total == 2
    assert pending["refresh"] == 2


# ---------------------------------------------------------------------------
# Tag-based routing (Phase B)
# ---------------------------------------------------------------------------


async def test_webhook_routes_ocr_tag_to_ocr_queue(webhook_session, task_queues, webhook_with_tags):
    """ai:run-ocr tag → queue:ocr."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/301/detail",
            "tag_list": "ai:run-ocr,invoice",
        },
    )
    assert r.status_code == 202
    assert 301 in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 301 not in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 301 not in _redis_stage_members(TaskQueues.KEY_EMBED)


async def test_webhook_routes_metadata_tag_to_metadata_queue(webhook_session, task_queues, webhook_with_tags):
    """ai:run-metadata tag → queue:metadata."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/302/detail",
            "tag_list": "ai:run-metadata",
        },
    )
    assert r.status_code == 202
    assert 302 in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 302 not in _redis_stage_members(TaskQueues.KEY_OCR)


async def test_webhook_routes_embed_tag_to_embed_queue(webhook_session, task_queues, webhook_with_tags):
    """ai:run-embed tag → queue:embed."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/303/detail",
            "tag_list": "ai:run-embed",
        },
    )
    assert r.status_code == 202
    assert 303 in _redis_stage_members(TaskQueues.KEY_EMBED)
    assert 303 not in _redis_stage_members(TaskQueues.KEY_OCR)


async def test_webhook_ignores_no_ai_tag(webhook_session, task_queues, webhook_with_tags):
    """No ai:run-* tag -> refresh queue only."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/304/detail",
            "tag_list": "invoice,personal",
        },
    )
    assert r.status_code == 202
    assert 304 not in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 304 not in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 304 not in _redis_stage_members(TaskQueues.KEY_EMBED)
    assert 304 in _redis_stage_members(TaskQueues.KEY_REFRESH)


async def test_webhook_ignores_missing_tags_field(webhook_session, task_queues, webhook_with_tags):
    """Missing tag_list key -> refresh queue only."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={"doc_url": "https://paperless.home/documents/305/detail"},
    )
    assert r.status_code == 202
    assert 305 not in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 305 not in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 305 not in _redis_stage_members(TaskQueues.KEY_EMBED)
    assert 305 in _redis_stage_members(TaskQueues.KEY_REFRESH)


async def test_webhook_enqueues_refresh_for_untagged_updates(
    webhook_session, task_queues, webhook_with_tags
):
    """Untagged updates enqueue refresh work without touching OCR/metadata/embed queues."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/308/detail",
            "tag_list": "invoice",
        },
    )
    assert r.status_code == 202
    assert 308 not in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 308 not in _redis_stage_members(TaskQueues.KEY_METADATA)
    assert 308 not in _redis_stage_members(TaskQueues.KEY_EMBED)
    assert 308 in _redis_stage_members(TaskQueues.KEY_REFRESH)


async def test_webhook_ocr_tag_takes_priority_over_embed(webhook_session, task_queues, webhook_with_tags):
    """If both ai:run-ocr and ai:run-embed are present, ocr wins."""
    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": "https://paperless.home/documents/306/detail",
            "tag_list": "ai:run-embed,ai:run-ocr",
        },
    )
    assert r.status_code == 202
    assert 306 in _redis_stage_members(TaskQueues.KEY_OCR)
    assert 306 not in _redis_stage_members(TaskQueues.KEY_EMBED)


async def test_webhook_uses_current_paperless_tags_over_payload_snapshot(
    webhook_session, task_queues, webhook_with_tags, paperless_client
):
    """Route using current Paperless tags when webhook payload tags are stale."""
    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_ocr_id = await paperless_client.get_tag_id("ai:run-ocr", create=True)
    await paperless_client.patch_document(doc_id, {"tags": [tag_ocr_id]})

    r = await webhook_session.post(
        f"{WEBHOOK_URL}/webhook/document",
        json={
            "doc_url": f"https://paperless.home/documents/{doc_id}/detail",
            "tag_list": "ai:run-embed",
        },
    )
    assert r.status_code == 202
    assert doc_id in _redis_stage_members(TaskQueues.KEY_OCR)
    assert doc_id not in _redis_stage_members(TaskQueues.KEY_EMBED)


# ---------------------------------------------------------------------------
# End-to-end: Paperless fires the webhook on document events
# ---------------------------------------------------------------------------

# Full webhook endpoint that the Paperless *webserver* container uses to reach
# the webhook-listener. This may differ from WEBHOOK_URL (used by the test
# runner / ai container) if the containers have different DNS views, but in the
# test compose they share the same `internal` network so the hostname resolves
# identically.
_PAPERLESS_FACING_WEBHOOK_ENDPOINT = os.environ.get(
    "PAPERLESS_WEBHOOK_URL",
    f"{WEBHOOK_URL}/webhook/document",
)

_TRIGGER_DOCUMENT_ADDED = 2
_TRIGGER_DOCUMENT_UPDATED = 3


@pytest.fixture
async def paperless_workflow(paperless_client):
    """
    Factory fixture: create a webhook workflow for each call, delete all on teardown.

    Usage::

        async def test_something(paperless_workflow):
            wf_id = await paperless_workflow(_TRIGGER_DOCUMENT_ADDED, "my-test-wf")
            ...
    """
    workflow_ids: list[int] = []

    async def _create(trigger_type: int, name: str, filter_has_tags: list | None = None) -> int:
        wf_id = await _create_webhook_workflow(paperless_client, trigger_type, name, filter_has_tags)
        workflow_ids.append(wf_id)
        return wf_id

    yield _create

    for wf_id in workflow_ids:
        await paperless_client._client.delete(f"/api/workflows/{wf_id}/")


async def _create_webhook_workflow(
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
                    "url": _PAPERLESS_FACING_WEBHOOK_ENDPOINT,
                    "use_params": True,
                    "as_json": True,
                    "params": {"doc_url": "{{doc_url}}"},
                    "headers": {"X-Webhook-Token": TEST_WEBHOOK_SECRET},
                },
            }
        ],
    }
    r = await client._client.post("/api/workflows/", json=payload)
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
    task_queues, paperless_workflow, uploaded_document
):
    """
    Paperless DOCUMENT_ADDED trigger → webhook → Redis.

    Creates a workflow with trigger type 2 (DOCUMENT_ADDED), uploads a
    document, and verifies the doc ID lands in the Redis queue.  Validates
    the full chain: workflow fires, {{doc_url}} renders, webhook-listener
    extracts the ID and enqueues it.
    """
    await paperless_workflow(_TRIGGER_DOCUMENT_ADDED, "test-wf-document-added")
    doc_id = await uploaded_document()
    assert await _wait_for_doc_in_queue(doc_id), (
        f"Document {doc_id} never appeared in the Redis queue within 60 s "
        f"(trigger=DOCUMENT_ADDED, webhook={_PAPERLESS_FACING_WEBHOOK_ENDPOINT})"
    )


async def test_paperless_fires_webhook_on_document_updated(
    paperless_client, task_queues, paperless_workflow, uploaded_document
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
    doc_id = await uploaded_document()
    await paperless_workflow(_TRIGGER_DOCUMENT_UPDATED, "test-wf-document-updated")

    r = await paperless_client._client.patch(
        f"/api/documents/{doc_id}/",
        json={"title": "Updated Title — DOCUMENT_UPDATED webhook test"},
    )
    assert r.status_code == 200, f"PATCH failed: {r.status_code} — {r.text}"
    assert await _wait_for_doc_in_queue(doc_id), (
        f"Document {doc_id} never appeared in the Redis queue within 60 s "
        f"(trigger=DOCUMENT_UPDATED, webhook={_PAPERLESS_FACING_WEBHOOK_ENDPOINT}). "
        "Check that trigger type 3 is DOCUMENT_UPDATED in this Paperless version."
    )


async def test_auto_managed_updated_workflow_routes_tagged_docs_to_ocr_queue(
    paperless_client, task_queues, uploaded_document
):
    """
    Real auto-managed workflow routing uses current Paperless tags for backfills.

    This covers the production path that regressed:
    1. Upload a document before workflows exist.
    2. Create/update the auto-managed workflows via ensure_ai_workflows().
    3. Add ai:run-ocr to the existing document.
    4. Verify Paperless fires DOCUMENT_UPDATED and the webhook listener routes
       the document to the OCR queue, not the embed queue.
    """
    doc_id = await uploaded_document()

    await paperless_client.ensure_ai_workflows(
        tag_ocr="ai:run-ocr",
        webhook_url=_PAPERLESS_FACING_WEBHOOK_ENDPOINT,
        webhook_secret=TEST_WEBHOOK_SECRET,
    )

    tag_id = await paperless_client.get_tag_id("ai:run-ocr", create=False)
    r = await paperless_client._client.patch(
        f"/api/documents/{doc_id}/",
        json={"tags": [tag_id]},
    )
    assert r.status_code == 200, f"Failed to add ai:run-ocr tag: {r.status_code} — {r.text}"

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if doc_id in _redis_stage_members(TaskQueues.KEY_OCR):
            break
        await asyncio.sleep(2)

    assert doc_id in _redis_stage_members(TaskQueues.KEY_OCR), (
        f"Document {doc_id} never appeared in OCR queue after adding ai:run-ocr"
    )
    assert doc_id not in _redis_stage_members(TaskQueues.KEY_EMBED), (
        f"Document {doc_id} incorrectly landed in embed queue instead of OCR queue"
    )


async def test_document_updated_deduplicates_repeated_edits(
    paperless_client, task_queues, paperless_workflow, uploaded_document
):
    """
    Multiple rapid edits to the same document produce exactly one queue entry.

    The webhook fires on each DOCUMENT_UPDATED event, but Redis SADD is
    idempotent — the pending set must contain the doc ID exactly once even
    after three consecutive PATCHes.
    """
    doc_id = await uploaded_document()
    await paperless_workflow(_TRIGGER_DOCUMENT_UPDATED, "test-wf-dedup-updates")

    for i in range(3):
        r = await paperless_client._client.patch(
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

