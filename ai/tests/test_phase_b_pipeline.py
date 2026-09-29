"""
Pipeline tests: TaskQueues, webhook routing, and the OCR and metadata batch
workers. These run against Paperless and Redis, with LLM and document OCR mocked.

Test matrix:
  TaskQueues — unit tests (fast, Redis required):
    test_task_queues_enqueue_and_peek
    test_task_queues_remove
    test_task_queues_pending_count
    test_task_queues_deduplication
    test_task_queues_mark_failure_moves_to_failed_queue

  Webhook routing — unit-style (no Paperless, Redis required):
    test_parse_tags_empty
    test_parse_tags_comma_separated
    test_route_to_stage_ocr
    test_route_to_stage_metadata

  run_ocr_batch — integration (Paperless + Redis + mock OCR):
    test_ocr_batch_writes_content_and_transitions_tag
    test_ocr_batch_skips_missing_document
    test_ocr_batch_dry_run
    test_ocr_batch_moves_poison_document_to_failed_queue

  run_metadata_batch — integration (Paperless + Redis + mock LLM):
    test_metadata_batch_writes_metadata_and_transitions_tag
    test_metadata_batch_skips_empty_content
    test_metadata_batch_dry_run

"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.usefixtures("mock_document_service")

from tests.conftest import (
    PAPERLESS_URL,
    _make_test_pdf,
    _upload_document,
)
from paperless_listener.app import _parse_tags, _route_to_stage
from paperless_common.queue import TaskQueues

# ---------------------------------------------------------------------------
# TaskQueues unit tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_task_queues_enqueue_and_peek(task_queues):
    """Enqueue to each stage and peek returns the right IDs."""
    await task_queues.enqueue_ocr(1)
    await task_queues.enqueue_metadata(2)

    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == {1}
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == {2}


@pytest.mark.requires_redis
async def test_task_queues_remove(task_queues):
    """Remove takes a doc out of the specified stage only."""
    await task_queues.enqueue_ocr(10)
    await task_queues.enqueue_metadata(10)

    await task_queues.remove(10, TaskQueues.KEY_OCR)

    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == {10}


@pytest.mark.requires_redis
async def test_task_queues_pending_count(task_queues):
    """pending_count returns per-stage dict."""
    await task_queues.enqueue_ocr(1)
    await task_queues.enqueue_ocr(2)
    await task_queues.enqueue_metadata(3)

    counts = await task_queues.pending_count()
    assert counts["ocr"] == 2
    assert counts["metadata"] == 1


@pytest.mark.requires_redis
async def test_task_queues_deduplication(task_queues):
    """Same doc_id enqueued twice stays as a single entry."""
    added1 = await task_queues.enqueue_metadata(42)
    added2 = await task_queues.enqueue_metadata(42)

    assert added1 is True
    assert added2 is False
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == {42}


@pytest.mark.requires_redis
async def test_task_queues_mark_failure_moves_to_failed_queue(task_queues):
    await task_queues.enqueue_ocr(55)

    retry_count, moved = await task_queues.mark_failure(
        55, TaskQueues.KEY_OCR, max_attempts=3, base_delay_seconds=60
    )
    assert (retry_count, moved) == (1, False)
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.peek_stage(TaskQueues.KEY_FAILED) == set()

    assert await task_queues.release_due(TaskQueues.KEY_OCR, now=0) == 0
    assert await task_queues.release_due(TaskQueues.KEY_OCR, now=9999999999) == 1
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == {55}

    retry_count, moved = await task_queues.mark_failure(
        55, TaskQueues.KEY_OCR, max_attempts=3, base_delay_seconds=60
    )
    assert (retry_count, moved) == (2, False)
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.release_due(TaskQueues.KEY_OCR, now=9999999999) == 1

    retry_count, moved = await task_queues.mark_failure(
        55, TaskQueues.KEY_OCR, max_attempts=3
    )
    assert (retry_count, moved) == (3, True)
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.peek_stage(TaskQueues.KEY_FAILED) == {55}


@pytest.mark.requires_redis
async def test_queue_inspection_does_not_release_due_retries(task_queues, monkeypatch):
    """Health and count reads leave due retries delayed until explicitly released."""
    from paperless_ai.search import webhook
    from paperless_listener import app as listener

    monkeypatch.setattr(webhook, "_queues", task_queues)
    monkeypatch.setattr(listener, "_queues", task_queues)
    for stage in (TaskQueues.KEY_OCR, TaskQueues.KEY_METADATA):
        await task_queues.mark_failure(55, stage, max_attempts=3, base_delay_seconds=0)
        assert await task_queues.peek_stage(stage) == set()
        assert await task_queues.stage_size(stage) == 0
        assert await task_queues.stage_work_count(stage) == 1

    assert await task_queues.pending_count() == {"ocr": 0, "metadata": 0}
    assert (await listener.health())["pending"] == {"ocr": 0, "metadata": 0}
    assert json.loads((await webhook.health()).body)["pending"] == {
        "ocr": 0,
        "metadata": 0,
    }
    for stage in (TaskQueues.KEY_OCR, TaskQueues.KEY_METADATA):
        assert await task_queues.release_due(stage) == 1
        assert await task_queues.stage_work_count(stage) == 1
        await task_queues.remove(55, stage)
        assert await task_queues.stage_work_count(stage) == 0


@pytest.mark.requires_redis
async def test_retry_promotion_cannot_resurrect_removed_work(task_queues, monkeypatch):
    """A removal between client-side selection and promotion must not be lost."""
    stage = TaskQueues.KEY_OCR
    await task_queues.mark_failure(55, stage, max_attempts=3, base_delay_seconds=0)
    select_due = task_queues._redis.zrangebyscore

    async def select_then_remove(*args, **kwargs):
        """Force the stale-selection race if selection leaves the Redis script."""
        due = await select_due(*args, **kwargs)
        await task_queues.remove(55, stage)
        return due

    monkeypatch.setattr(task_queues._redis, "zrangebyscore", select_then_remove)
    await asyncio.gather(task_queues.release_due(stage), task_queues.remove(55, stage))
    assert await task_queues.stage_work_count(stage) == 0
    assert await task_queues.release_due(stage) == 0

    await task_queues.mark_failure(56, stage, max_attempts=3, base_delay_seconds=0)
    released = await asyncio.gather(
        task_queues.release_due(stage), task_queues.release_due(stage)
    )
    assert sum(released) == 1
    assert await task_queues.peek_stage(stage) == {56}


@pytest.mark.requires_redis
@pytest.mark.parametrize("stage", [TaskQueues.KEY_OCR, TaskQueues.KEY_METADATA])
async def test_batches_release_due_retries(task_queues, stage):
    """A batch promotes due retries while future retries still prevent draining."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch, run_metadata_batch

    client = AsyncMock()
    client.get_document.return_value = None
    client.get_document_with_content.return_value = None
    config = AgentConfig(ocr_endpoint="http://document-service.invalid")
    await task_queues.mark_failure(55, stage, max_attempts=3, base_delay_seconds=0)
    await task_queues.mark_failure(56, stage, max_attempts=3, base_delay_seconds=3600)

    if stage == TaskQueues.KEY_OCR:
        assert await run_ocr_batch(client, config, task_queues, 1) == (0, 0)
        client.get_document.assert_awaited_once_with(55)
    else:
        assert await run_metadata_batch(client, config, task_queues, 1, 2, 3) == (0, 0)
        client.get_document_with_content.assert_awaited_once_with(55)
    assert await task_queues.stage_size(stage) == 0
    assert await task_queues.stage_work_count(stage) == 1


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


def test_route_to_stage_without_processing_tag():
    assert _route_to_stage({"invoice", "personal"}) is None
    assert _route_to_stage(set()) is None


# ---------------------------------------------------------------------------
# run_ocr_batch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_ocr_batch_writes_content_and_transitions_tag(
    paperless_client, task_queues
):
    """
    OCR batch: downloads PDF, runs document OCR (mocked), writes content to
    Paperless, transitions tag from ai:run-ocr to ai:run-metadata, enqueues
    to metadata queue, removes from OCR queue.
    """
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
        tag_ocr="ai:run-ocr",
        tag_metadata="ai:run-metadata",
    )

    # Upload document and add ai:run-ocr tag
    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    tag_ocr_id = await paperless_client.get_tag_id(config.tag_ocr, create=True)
    tag_metadata_id = await paperless_client.get_tag_id(
        config.tag_metadata, create=True
    )
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
async def test_ocr_batch_skips_missing_document(paperless_client, task_queues):
    """Non-existent doc ID is silently removed from OCR queue (no crash)."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
    )

    await task_queues.enqueue_ocr(999999)
    success, failure = await run_ocr_batch(paperless_client, config, task_queues)
    assert success == 0
    assert failure == 0  # Missing docs are silently removed, not counted as failures
    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()


@pytest.mark.requires_redis
async def test_ocr_batch_dry_run(paperless_client, task_queues):
    """Dry-run OCR batch: returns success but does not modify Paperless."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
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


@pytest.mark.requires_redis
async def test_ocr_batch_moves_poison_document_to_failed_queue(
    paperless_client, task_queues, monkeypatch
):
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_ocr_batch

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("corrupted pdf")

    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.run_paddle_ocr", _boom)

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
        stage_max_attempts=3,
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    await task_queues.enqueue_ocr(doc_id)

    for attempt in range(3):
        success, failure = await run_ocr_batch(paperless_client, config, task_queues)
        assert success == 0
        assert failure == 1
        if attempt < 2:
            assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
            assert (
                await task_queues.release_due(TaskQueues.KEY_OCR, now=9999999999) == 1
            )

    assert await task_queues.peek_stage(TaskQueues.KEY_OCR) == set()
    assert await task_queues.peek_stage(TaskQueues.KEY_FAILED) == {doc_id}

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


# ---------------------------------------------------------------------------
# run_metadata_batch integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_redis
async def test_metadata_batch_writes_metadata_and_transitions_tag(
    paperless_client, task_queues, monkeypatch
):
    """
    Metadata batch: reads content from Paperless, runs LLM (mocked), writes
    title/date/correspondent/custom_fields and removes the metadata tag.
    """
    from datetime import date
    import json
    from unittest.mock import AsyncMock

    from paperless_ai.agents.smart_graph_agent import (
        StructuredOutputStrategy,
        _ExtractedMetadata,
    )
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
        tag_metadata="ai:run-metadata",
    )
    monkeypatch.setattr(
        StructuredOutputStrategy,
        "extract",
        AsyncMock(
            return_value=_ExtractedMetadata(
                title="Test Invoice",
                date=date(2024, 1, 15),
                correspondent="Acme Corp",
                summary="Invoice from Acme Corp dated 2024-01-15 for $100.00.",
                languages=["de", "en"],
            )
        ),
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
    tag_metadata_id = await paperless_client.get_tag_id(
        config.tag_metadata, create=True
    )
    stale_language_id = await paperless_client.get_tag_id("language:fr")
    unrelated_tag_id = await paperless_client.get_tag_id("integration:preserved")
    original_content = (
        "INVOICE\nAcme Corp\n123 Main St\nDate: January 15, 2024\n"
        "Please pay the total amount of $100.00 within 30 days.\n"
        "RECHNUNG\nBitte bezahlen Sie den Gesamtbetrag von $100.00 innerhalb "
        "von 30 Tagen. Vielen Dank für Ihren Auftrag."
    )

    await paperless_client.patch_document(
        doc_id,
        {
            "content": original_content,
            "tags": [tag_metadata_id, stale_language_id, unrelated_tag_id],
        },
    )
    # Cache pre-extraction counts; inventory must refresh after the document PATCH.
    await paperless_client._get_all_tags(force=True)
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client,
        config,
        task_queues,
        custom_field_id,
        ai_summary_field_id,
        ai_result_field_id,
    )
    assert success == 1, f"Expected 1 success, got {success=} {failure=}"
    assert failure == 0

    # Metadata queue is drained
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == set()

    # Fetch updated doc
    doc = await paperless_client.get_document_with_content(doc_id)

    # Title updated
    assert doc["title"] == "Test Invoice"

    # ai_processed custom field set to today
    cf_map = {cf["field"]: cf["value"] for cf in doc.get("custom_fields", [])}
    assert custom_field_id in cf_map
    assert cf_map[custom_field_id] == date.today().isoformat()
    assert (
        cf_map[ai_summary_field_id]
        == "Invoice from Acme Corp dated 2024-01-15 for $100.00."
    )
    assert json.loads(cf_map[ai_result_field_id])["ai_metadata"]["languages"] == ["de"]
    assert doc["content"] == original_content

    # Replace stale language tags while keeping unrelated tags.
    assert tag_metadata_id not in doc["tags"]
    assert stale_language_id not in doc["tags"]
    assert unrelated_tag_id in doc["tags"]
    tags = await paperless_client._get_all_tags(force=True)
    languages = {
        tag["name"]: tag
        for tag in tags
        if tag["name"].startswith("language:") and tag["document_count"] > 0
    }
    assert "language:fr" not in languages
    for code in ("de",):
        tag = languages[f"language:{code}"]
        assert tag["id"] in doc["tags"]
        assert tag["matching_algorithm"] == 0
        assert tag["color"] == "#166534"
        response = await paperless_client._client.get(
            "/api/documents/",
            params={"tags__id__in": tag["id"], "id": doc_id},
        )
        response.raise_for_status()
        assert doc_id in {result["id"] for result in response.json()["results"]}

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_metadata_batch_skips_empty_content(paperless_client, task_queues):
    """Document with no content is removed from metadata queue without processing."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
    )

    custom_field_id = await paperless_client.get_or_create_custom_field("ai_processed")
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    # Leave content empty (default after upload)
    await paperless_client.patch_document(doc_id, {"content": ""})
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client,
        config,
        task_queues,
        custom_field_id,
        ai_summary_field_id,
        ai_result_field_id,
    )
    assert success == 0
    assert failure == 0
    assert await task_queues.peek_stage(TaskQueues.KEY_METADATA) == set()

    await paperless_client._client.delete(f"/api/documents/{doc_id}/")


@pytest.mark.requires_redis
async def test_metadata_batch_dry_run(paperless_client, task_queues):
    """Dry-run metadata batch: returns success but does not modify Paperless."""
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.runner import run_metadata_batch

    token = paperless_client._client.headers["Authorization"].split(" ")[1]
    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=token,
        metadata_model="gemini/gemini-2.5-flash",
        chat_model="gemini/gemini-2.5-flash",
        dry_run=True,
    )

    custom_field_id = await paperless_client.get_or_create_custom_field("ai_processed")
    ai_summary_field_id = await paperless_client.get_or_create_custom_field(
        "ai_summary", data_type="longtext"
    )
    ai_result_field_id = await paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    doc_id = await _upload_document(paperless_client, _make_test_pdf())
    original = await paperless_client._client.get(f"/api/documents/{doc_id}/")
    original_title = original.json()["title"]

    await paperless_client.patch_document(
        doc_id, {"content": "INVOICE\nAcme Corp\n123 Main St"}
    )
    await task_queues.enqueue_metadata(doc_id)

    success, failure = await run_metadata_batch(
        paperless_client,
        config,
        task_queues,
        custom_field_id,
        ai_summary_field_id,
        ai_result_field_id,
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
