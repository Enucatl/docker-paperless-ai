"""
Phase B pipeline tests: TaskQueues, webhook routing, and the three decoupled
batch workers (run_ocr_batch, run_metadata_batch, run_embed_batch).

These are integration tests that run against a real Paperless instance, Redis,
and Qdrant (with LiteLLM and vision OCR mocked deterministically).

Test matrix:
  TaskQueues — unit tests (fast, Redis required):
    test_task_queues_enqueue_and_peek
    test_task_queues_remove
    test_task_queues_pending_count
    test_task_queues_deduplication

  Webhook routing — unit-style (no Paperless, Redis required):
    test_parse_tags_empty
    test_parse_tags_comma_separated
    test_route_to_stage_ocr
    test_route_to_stage_metadata
    test_route_to_stage_embed_explicit
    test_route_to_stage_embed_fallback

  run_ocr_batch — integration (Paperless + Redis + mock OCR):
    test_ocr_batch_writes_content_and_transitions_tag
    test_ocr_batch_skips_missing_document
    test_ocr_batch_dry_run

  run_metadata_batch — integration (Paperless + Redis + mock LLM):
    test_metadata_batch_writes_metadata_and_transitions_tag
    test_metadata_batch_skips_empty_content
    test_metadata_batch_dry_run

  run_embed_batch — integration (Paperless + Redis + mock embedder + Qdrant):
    test_embed_batch_upserts_qdrant_and_removes_tag
    test_embed_batch_skips_empty_content
    test_embed_batch_dry_run

  Full three-stage flow:
    test_full_phase_b_pipeline_sequential
"""

import os

import pytest

from tests.conftest import (
    PAPERLESS_URL,
    REDIS_URL,
    _redis_stage_members,
    _make_test_pdf,
    _upload_document,
)
from paperless_ai.search.queue import TaskQueues
from paperless_ai.search.webhook import _parse_tags, _route_to_stage

# ---------------------------------------------------------------------------
# TaskQueues unit tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_task_queues_enqueue_and_peek(task_queues):
    """Enqueue to each stage and peek returns the right IDs."""
    await task_queues.enqueue_ocr(1)
    await task_queues.enqueue_metadata(2)
    await task_queues.enqueue_embed(3)

    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == {1}
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == {2}
    assert await task_queues.peek_stage(TaskQueues.KEY_EMBED) == {3}


@pytest.mark.requires_redis
async def test_task_queues_remove(task_queues):
    """Remove takes a doc out of the specified stage only."""
    await task_queues.enqueue_ocr(10)
    await task_queues.enqueue_embed(10)

    await task_queues.remove(10, TaskQueues.KEY_OCR)

    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.peek_stage(TaskQueues.KEY_EMBED) == {10}


@pytest.mark.requires_redis
async def test_task_queues_pending_count(task_queues):
    """pending_count returns per-stage dict."""
    await task_queues.enqueue_ocr(1)
    await task_queues.enqueue_ocr(2)
    await task_queues.enqueue_metadata(3)

    counts = await task_queues.pending_count()
    assert counts["ocr"] == 2
    assert counts["metadata"] == 1
    assert counts["embed"] == 0


@pytest.mark.requires_redis
async def test_task_queues_deduplication(task_queues):
    """Same doc_id enqueued twice stays as a single entry."""
    added1 = await task_queues.enqueue_embed(42)
    added2 = await task_queues.enqueue_embed(42)

    assert added1 is True
    assert added2 is False
    assert await task_queues.peek_stage(TaskQueues.KEY_EMBED) == {42}


# ---------------------------------------------------------------------------
# Webhook routing unit tests (pure Python — no HTTP, no Paperless)
# ---------------------------------------------------------------------------


def test_parse_tags_empty():
    assert _parse_tags({}) == set()
    assert _parse_tags({"tag_list": ""}) == set()


def test_parse_tags_comma_separated():
    body = {"tag_list": "ai:run-ocr, invoice, personal"}
    assert _parse_tags(body) == {"ai:run-ocr", "invoice", "personal"}


def test_route_to_stage_ocr():
    assert _route_to_stage({"ai:run-ocr"}) == TaskQueues.KEY_OCR


def test_route_to_stage_metadata():
    assert _route_to_stage({"ai:run-metadata"}) == TaskQueues.KEY_METADATA


def test_route_to_stage_embed_explicit():
    assert _route_to_stage({"ai:run-embed"}) == TaskQueues.KEY_EMBED


def test_route_to_stage_embed_fallback():
    """No ai:run-* tag → embed queue (human edit)."""
    assert _route_to_stage({"invoice", "personal"}) == TaskQueues.KEY_EMBED
    assert _route_to_stage(set()) == TaskQueues.KEY_EMBED


def test_route_to_stage_ocr_priority_over_embed():
    """OCR tag takes priority when multiple ai:run-* tags are present."""
    assert _route_to_stage({"ai:run-ocr", "ai:run-embed"}) == TaskQueues.KEY_OCR


def test_route_to_stage_metadata_priority_over_embed():
    assert _route_to_stage({"ai:run-metadata", "ai:run-embed"}) == TaskQueues.KEY_METADATA


# ---------------------------------------------------------------------------
# run_ocr_batch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_ocr_batch_writes_content_and_transitions_tag(
    paperless_client, task_queues
):
    """
    OCR batch: downloads PDF, runs vision OCR (mocked), writes content to
    Paperless, transitions tag from ai:run-ocr to ai:run-metadata, enqueues
    to metadata queue, removes from OCR queue.
    """
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        ocr_model="gemini/gemini-2.5-flash",
        tag_ocr="ai:run-ocr",
        tag_metadata="ai:run-metadata",
        tag_embed="ai:run-embed",
    )

    # Upload document and add ai:run-ocr tag
    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_ocr_id = await paperless_client.get_tag_id(config.tag_ocr, create=True)
    tag_metadata_id = await paperless_client.get_tag_id(config.tag_metadata, create=True)
    await paperless_client.patch_document(doc_id, {"tags": [tag_ocr_id]})

    # Enqueue to OCR queue
    await task_queues.enqueue_ocr(doc_id)

    # Act
    success, failure = await run_ocr_batch(paperless_client, config, task_queues)
    assert success == 1, f"Expected 1 success, got {success=} {failure=}"
    assert failure == 0

    # OCR queue is drained
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()

    # Metadata queue received the doc
    assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_METADATA)

    # Paperless content is updated
    doc = await paperless_client.get_document_with_content(doc_id)
    assert doc["content"].strip(), "Content should have been written by OCR batch"

    # Tag transitioned: ai:run-ocr removed, ai:run-metadata added
    assert tag_ocr_id not in doc["tags"]
    assert tag_metadata_id in doc["tags"]

    # Cleanup
    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_ocr_batch_skips_missing_document(paperless_client, task_queues, document_queue):
    """Non-existent doc ID is silently removed from OCR queue (no crash)."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(paperless_url=PAPERLESS_URL, paperless_token=token)

    await task_queues.enqueue_ocr(999999)
    success, failure = await run_ocr_batch(paperless_client, config, task_queues)
    assert success == 0
    assert failure == 0  # Missing docs are silently removed, not counted as failures
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()


@pytest.mark.requires_redis
async def test_ocr_batch_dry_run(paperless_client, task_queues, document_queue):
    """Dry-run OCR batch: returns success but does not modify Paperless."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        dry_run=True,
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    original = await paperless_client.get_document_with_content(doc_id)
    original_content = original.get("content", "")

    await task_queues.enqueue_ocr(doc_id)
    success, failure = await run_ocr_batch(paperless_client, config, task_queues)
    assert success == 1
    assert failure == 0

    # Queue is NOT drained in dry-run
    assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_OCR)

    # Content unchanged
    after = await paperless_client.get_document_with_content(doc_id)
    assert after.get("content", "") == original_content

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


# ---------------------------------------------------------------------------
# run_metadata_batch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_metadata_batch_writes_metadata_and_transitions_tag(
    paperless_client, task_queues
):
    """
    Metadata batch: reads content from Paperless, runs LLM (mocked), writes
    title/date/correspondent/custom_fields, transitions tag to ai:run-embed,
    enqueues to embed queue.
    """
    from datetime import date

    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        tag_metadata="ai:run-metadata",
        tag_embed="ai:run-embed",
    )

    custom_field_id = await paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    # Upload doc, write content (as if OCR stage already ran)
    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_metadata_id = await paperless_client.get_tag_id(config.tag_metadata, create=True)
    tag_embed_id = await paperless_client.get_tag_id(config.tag_embed, create=True)

    await paperless_client.patch_document(
        doc_id,
        {
            "content": "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024",
            "tags": [tag_metadata_id],
        },
    )
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client, config, task_queues, custom_field_id, ai_summary_field_id, ai_result_field_id
    )
    assert success == 1, f"Expected 1 success, got {success=} {failure=}"
    assert failure == 0

    # Metadata queue is drained
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == set()

    # Embed queue received the doc
    assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_EMBED)

    # Fetch updated doc
    doc = await paperless_client.get_document_with_content(doc_id)

    # Title updated
    assert doc["title"] == "Test Invoice"

    # ai_processed custom field set to today
    cf_map = {cf["field"]: cf["value"] for cf in doc.get("custom_fields", [])}
    assert custom_field_id in cf_map
    assert cf_map[custom_field_id] == date.today().isoformat()
    assert cf_map[ai_summary_field_id] == "Invoice from Acme Corp dated 2024-01-15 for $100.00."

    # Tag transitioned: ai:run-metadata removed, ai:run-embed added
    assert tag_metadata_id not in doc["tags"]
    assert tag_embed_id in doc["tags"]

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_metadata_batch_skips_empty_content(paperless_client, task_queues, document_queue):
    """Document with no content is removed from metadata queue without processing."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(paperless_url=PAPERLESS_URL, paperless_token=token)

    custom_field_id = await paperless_client.get_or_create_custom_field("ai_processed")
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field("ai_result", data_type="longtext")

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    # Leave content empty (default after upload)
    await paperless_client.patch_document(doc_id, {"content": ""})
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client, config, task_queues, custom_field_id, ai_summary_field_id, ai_result_field_id
    )
    assert success == 0
    assert failure == 0
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == set()

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_metadata_batch_dry_run(paperless_client, task_queues, document_queue):
    """Dry-run metadata batch: returns success but does not modify Paperless."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        dry_run=True,
    )

    custom_field_id = await paperless_client.get_or_create_custom_field("ai_processed")
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field("ai_result", data_type="longtext")

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    original = await paperless_client._client.get(f"/api/documents/{doc_id}/")
    original_title = original.json()["title"]

    await paperless_client.patch_document(
        doc_id, {"content": "INVOICE\nAcme Corp\n123 Main St"}
    )
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client, config, task_queues, custom_field_id, ai_summary_field_id, ai_result_field_id
    )
    assert success == 1
    assert failure == 0

    # Queue NOT drained in dry-run
    assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_METADATA)

    # Title unchanged
    after = await paperless_client._client.get(f"/api/documents/{doc_id}/")
    assert after.json()["title"] == original_title

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


# ---------------------------------------------------------------------------
# run_embed_batch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_embed_batch_upserts_qdrant_and_removes_tag(
    paperless_client, task_queues, mock_embedder, qdrant_store
):
    """
    Embed batch: reads content + metadata from Paperless, embeds (mock),
    upserts Qdrant, removes ai:run-embed tag from document.
    """
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_embed_batch
    from paperless_ai.search.qdrant_store import COLLECTION
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        tag_embed="ai:run-embed",
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_embed_id = await paperless_client.get_tag_id(config.tag_embed, create=True)

    await paperless_client.patch_document(
        doc_id,
        {
            "title": "Test Invoice",
            "content": "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024",
            "tags": [tag_embed_id],
        },
    )
    await task_queues.enqueue_embed(doc_id)

    success, failure = await run_embed_batch(
        paperless_client, config, task_queues, qdrant_store, mock_embedder
    )
    assert success == 1, f"Expected 1 success, got {success=} {failure=}"
    assert failure == 0

    # Embed queue is drained
    assert await task_queues.peek_stage(TaskQueues.KEY_EMBED) == set()

    # Vectors exist in Qdrant
    results, _ = await qdrant_store._client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=100,
    )
    assert len(results) > 0, f"No Qdrant vectors for doc_id={doc_id}"
    assert results[0].payload["title"] == "Test Invoice"

    # ai:run-embed tag removed from document
    doc = await paperless_client.get_document(doc_id)
    assert tag_embed_id not in doc["tags"]

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_embed_batch_skips_empty_content(paperless_client, task_queues, qdrant_store, document_queue):
    """Document with empty content is processed (tag removed) but nothing embedded."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_embed_batch
    from paperless_ai.search.qdrant_store import COLLECTION
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        tag_embed="ai:run-embed",
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_embed_id = await paperless_client.get_tag_id(config.tag_embed, create=True)
    await paperless_client.patch_document(doc_id, {"content": "", "tags": [tag_embed_id]})
    await task_queues.enqueue_embed(doc_id)

    success, failure = await run_embed_batch(paperless_client, config, task_queues, qdrant_store, None)
    assert success == 1
    assert failure == 0
    assert await task_queues.peek_stage(TaskQueues.KEY_EMBED) == set()

    # No vectors in Qdrant (content was empty)
    results, _ = await qdrant_store._client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=10,
    )
    assert len(results) == 0

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_embed_batch_dry_run(paperless_client, task_queues, mock_embedder, qdrant_store, document_queue):
    """Dry-run embed batch: returns success but does not remove tag or modify Qdrant."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_embed_batch
    from paperless_ai.search.qdrant_store import COLLECTION
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        tag_embed="ai:run-embed",
        dry_run=True,
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_embed_id = await paperless_client.get_tag_id(config.tag_embed, create=True)
    await paperless_client.patch_document(
        doc_id,
        {"content": "INVOICE\nAcme Corp", "tags": [tag_embed_id]},
    )
    await task_queues.enqueue_embed(doc_id)

    success, failure = await run_embed_batch(
        paperless_client, config, task_queues, qdrant_store, mock_embedder
    )
    assert success == 1
    assert failure == 0

    # Queue NOT drained in dry-run
    assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_EMBED)

    # Tag still present
    doc = await paperless_client.get_document(doc_id)
    assert tag_embed_id in doc["tags"]

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


# ---------------------------------------------------------------------------
# Full three-stage flow
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_full_phase_b_pipeline_sequential(
    paperless_client, task_queues, mock_embedder, qdrant_store
):
    """
    Full Phase B pipeline: doc enters OCR queue, flows through all three stages
    sequentially (--once mode), ends up embedded in Qdrant with all three tags
    removed.

    LiteLLM is mocked by the session-scoped mock_litellm fixture.
    """
    from datetime import date

    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_embed_batch, run_metadata_batch, run_ocr_batch
    from paperless_ai.search.qdrant_store import COLLECTION
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        tag_ocr="ai:run-ocr",
        tag_metadata="ai:run-metadata",
        tag_embed="ai:run-embed",
    )

    custom_field_id = await paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    # Upload document and add ai:run-ocr tag (as Paperless workflow would do)
    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_ocr_id = await paperless_client.get_tag_id(config.tag_ocr, create=True)
    tag_metadata_id = await paperless_client.get_tag_id(config.tag_metadata, create=True)
    tag_embed_id = await paperless_client.get_tag_id(config.tag_embed, create=True)
    await paperless_client.patch_document(doc_id, {"tags": [tag_ocr_id]})

    await task_queues.enqueue_ocr(doc_id)

    # Stage 1: OCR
    ocr_s, ocr_f = await run_ocr_batch(paperless_client, config, task_queues)
    assert ocr_s == 1 and ocr_f == 0, f"OCR stage failed: {ocr_s=} {ocr_f=}"

    # Stage 2: Metadata
    meta_s, meta_f = await run_metadata_batch(
        paperless_client, config, task_queues, custom_field_id, ai_summary_field_id, ai_result_field_id
    )
    assert meta_s == 1 and meta_f == 0, f"Metadata stage failed: {meta_s=} {meta_f=}"

    # Stage 3: Embed
    embed_s, embed_f = await run_embed_batch(
        paperless_client, config, task_queues, qdrant_store, mock_embedder
    )
    assert embed_s == 1 and embed_f == 0, f"Embed stage failed: {embed_s=} {embed_f=}"

    # All queues empty
    counts = await task_queues.pending_count()
    assert sum(counts.values()) == 0, f"Queues not fully drained: {counts}"

    # Fetch final document state
    doc = await paperless_client.get_document_with_content(doc_id)

    # Content written by OCR stage
    assert doc["content"].strip(), "Content should have been written"

    # Metadata written by metadata stage
    assert doc["title"] == "Test Invoice"
    cf_map = {cf["field"]: cf["value"] for cf in doc.get("custom_fields", [])}
    assert custom_field_id in cf_map
    assert cf_map[custom_field_id] == date.today().isoformat()
    assert cf_map[ai_summary_field_id] == "Invoice from Acme Corp dated 2024-01-15 for $100.00."

    # All ai:run-* tags removed
    for tag_id in [tag_ocr_id, tag_metadata_id, tag_embed_id]:
        assert tag_id not in doc["tags"], (
            f"Tag id={tag_id} should have been removed from document"
        )

    # Vectors exist in Qdrant
    results, _ = await qdrant_store._client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=100,
    )
    assert len(results) > 0, f"No Qdrant vectors for doc_id={doc_id}"

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")
