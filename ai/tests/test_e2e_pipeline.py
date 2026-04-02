"""
E2E pipeline tests: exercise the full document processing flow against a
real (ephemeral) Paperless-ngx instance and Redis queue, with LiteLLM
calls mocked deterministically.

Test matrix:
  test_full_pipeline_patches_document_correctly
    — Redis queue → OCR → metadata → Paperless PATCH → queue drained
  test_dry_run_does_not_modify_document
    — dry_run=True → Paperless unchanged, queue NOT drained
  test_pipeline_embeds_into_qdrant
    — full pipeline with mock embedder → vectors appear in Qdrant
"""

import os

import pytest

from tests.conftest import PAPERLESS_URL, _redis_queue_size

PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")


async def test_full_pipeline_patches_document_correctly(
    paperless_client, dummy_document: int, document_queue
):
    """
    Full path: Redis queue → SmartDocumentAgent (OCR + metadata) → Paperless PATCH
    → SREM from queue.

    Asserts:
    - Paperless document title, date, correspondent are updated
    - ai_processed custom field is set to today's date
    - Redis queue is empty after processing
    """
    from datetime import date

    from agents.smart_graph_agent import SmartDocumentAgent, _select_extraction_strategy
    from core.config import AgentConfig
    from core.runner import run_batch

    doc_id = dummy_document

    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=paperless_client._client.headers["Authorization"].split(" ")[1],
        ocr_model="gemini/gemini-2.5-flash",
        dry_run=False,
    )

    agent = SmartDocumentAgent(config, extraction_strategy=_select_extraction_strategy(config))

    custom_field_id = paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_result_field_id = paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    # Act — no store/embedder: embedding step is skipped
    success, failure = await run_batch(
        paperless_client, agent, config, custom_field_id, ai_result_field_id,
        document_queue,
    )

    assert success == 1, f"Expected 1 success, got success={success} failure={failure}"
    assert failure == 0, f"Expected 0 failures, got failure={failure}"

    # Redis queue must be empty — doc was SREM'd on success
    assert _redis_queue_size() == 0, "Queue should be drained after successful processing"

    # Fetch the updated document from Paperless
    r = paperless_client._client.get(f"/api/documents/{doc_id}/")
    r.raise_for_status()
    doc = r.json()

    # Title was updated to the mocked value
    assert doc["title"] == "Test Invoice", f"Unexpected title: {doc['title']!r}"

    # ai_processed custom field was set to today's date
    cf_map = {cf["field"]: cf["value"] for cf in doc.get("custom_fields", [])}
    assert custom_field_id in cf_map, (
        f"ai_processed custom field (id={custom_field_id}) not found; fields={cf_map}"
    )
    assert cf_map[custom_field_id] == date.today().isoformat(), (
        f"ai_processed date mismatch: {cf_map[custom_field_id]!r}"
    )

    # Correspondent was created and linked
    assert doc.get("correspondent") is not None, "Correspondent should have been set"
    correspondent_name = paperless_client.get_correspondent_name(doc["correspondent"])
    assert correspondent_name == "Acme Corp", (
        f"Expected correspondent 'Acme Corp', got {correspondent_name!r}"
    )


async def test_dry_run_does_not_modify_document(
    paperless_client, dummy_document: int, document_queue
):
    """
    In dry-run mode, run_batch must return success=1 but leave the document
    completely untouched and the Redis queue intact.
    """
    from agents.smart_graph_agent import SmartDocumentAgent, _select_extraction_strategy
    from core.config import AgentConfig
    from core.runner import run_batch

    doc_id = dummy_document

    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=paperless_client._client.headers["Authorization"].split(" ")[1],
        ocr_model="gemini/gemini-2.5-flash",
        dry_run=True,  # ← dry run
    )

    agent = SmartDocumentAgent(config, extraction_strategy=_select_extraction_strategy(config))

    custom_field_id = paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_result_field_id = paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    # Snapshot before
    r_before = paperless_client._client.get(f"/api/documents/{doc_id}/")
    r_before.raise_for_status()
    doc_before = r_before.json()

    success, failure = await run_batch(
        paperless_client, agent, config, custom_field_id, ai_result_field_id,
        document_queue,
    )

    assert success == 1
    assert failure == 0

    # Queue must NOT be drained — dry-run never calls SREM
    assert _redis_queue_size() == 1, (
        "Dry-run must not remove the document from the Redis queue"
    )

    # Fetch again — nothing should have changed
    r_after = paperless_client._client.get(f"/api/documents/{doc_id}/")
    r_after.raise_for_status()
    doc_after = r_after.json()

    assert doc_after["title"] == doc_before["title"], "Dry-run must not change title"
    cf_map = {cf["field"]: cf["value"] for cf in doc_after.get("custom_fields", [])}
    assert custom_field_id not in cf_map, "Dry-run must not set ai_processed field"


async def test_pipeline_embeds_into_qdrant(
    paperless_client, dummy_document: int, document_queue, mock_embedder, qdrant_store
):
    """
    Full pipeline with embedding: OCR + metadata + mock embed → vectors appear in Qdrant.

    Uses mock_embedder (deterministic fake vectors) because the GPU-hosted
    Infinity server is not available in the test environment.
    """
    from agents.smart_graph_agent import SmartDocumentAgent, _select_extraction_strategy
    from core.config import AgentConfig
    from core.runner import run_batch
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    doc_id = dummy_document

    config = AgentConfig(
        paperless_url=PAPERLESS_URL,
        paperless_token=paperless_client._client.headers["Authorization"].split(" ")[1],
        ocr_model="gemini/gemini-2.5-flash",
        dry_run=False,
    )

    agent = SmartDocumentAgent(config, extraction_strategy=_select_extraction_strategy(config))

    custom_field_id = paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_result_field_id = paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    success, failure = await run_batch(
        paperless_client, agent, config, custom_field_id, ai_result_field_id,
        document_queue,
        store=qdrant_store,
        embedder=mock_embedder,
    )

    assert success == 1, f"Expected 1 success, got success={success} failure={failure}"
    assert failure == 0

    # Vectors for this document must exist in Qdrant
    results, _ = await qdrant_store._client.scroll(
        collection_name=qdrant_store.COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
        limit=100,
    )
    assert len(results) > 0, (
        f"No Qdrant vectors found for doc_id={doc_id} after pipeline run"
    )

    # Each point must carry the expected payload fields
    for point in results:
        p = point.payload
        assert p["doc_id"] == doc_id
        assert p["title"] == "Test Invoice"
        assert p["correspondent"] == "Acme Corp"
        assert "text" in p
