"""
Integration tests for the webhook listener service.

These tests POST to the live webhook-listener container
(http://webhook-listener:8001) and verify that document IDs land in the
Redis queue (DB 1) as expected.

The webhook-listener service must be running before pytest starts — this is
guaranteed by the `ai` service's depends_on in docker-compose.test.yml.
"""

import os

import httpx
import pytest

from tests.conftest import (
    WEBHOOK_URL,
    _redis_queue_members,
    _redis_queue_size,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_webhook_health(document_queue):
    """GET /health returns 200 with a pending count."""
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{WEBHOOK_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "pending" in body


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
# Pending count reflected in health endpoint
# ---------------------------------------------------------------------------


async def test_webhook_health_reflects_pending_count(document_queue):
    """
    After enqueuing two documents, /health must report pending=2.
    """
    payloads = [
        {"doc_url": "https://paperless.home/documents/201/detail"},
        {"doc_url": "https://paperless.home/documents/202/detail"},
    ]
    async with httpx.AsyncClient() as client:
        for p in payloads:
            await client.post(f"{WEBHOOK_URL}/webhook/document", json=p)

        r = await client.get(f"{WEBHOOK_URL}/health")

    assert r.json()["pending"] == 2
