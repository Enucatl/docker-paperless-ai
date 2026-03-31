"""
E2E pipeline test: exercises the full document processing flow against a
real (ephemeral) Paperless-ngx instance, with only LiteLLM calls mocked.

Verifies that:
1. The agent downloads the document, OCRs/extracts metadata, and PATCHes Paperless.
2. The document title is updated to the mocked title.
3. The ai-review-pending tag is removed.
4. The ai_processed custom field is set to today's date.
5. The correspondent "Acme Corp" is created in Paperless and linked to the document.
"""

import os
from datetime import date

import pytest

PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")


@pytest.mark.asyncio
async def test_full_pipeline_patches_document_correctly(
    paperless_client, dummy_document: int
):
    """
    Full path: SmartDocumentAgent → native text extraction → metadata extraction
    → Paperless PATCH. Asserts all fields are written and the pending tag removed.
    """
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

    pending_id = paperless_client.get_tag_id(config.tag_pending, create=True)
    custom_field_id = paperless_client.get_or_create_custom_field(
        "ai_processed", data_type="date"
    )
    ai_result_field_id = paperless_client.get_or_create_custom_field(
        "ai_result", data_type="longtext"
    )

    # Act
    success, failure = await run_batch(
        paperless_client, agent, config, pending_id, custom_field_id, ai_result_field_id
    )

    assert success == 1, f"Expected 1 success, got success={success} failure={failure}"
    assert failure == 0, f"Expected 0 failures, got failure={failure}"

    # Fetch the updated document from Paperless
    r = paperless_client._client.get(f"/api/documents/{doc_id}/")
    r.raise_for_status()
    doc = r.json()

    # Title was updated to the mocked value
    assert doc["title"] == "Test Invoice", f"Unexpected title: {doc['title']!r}"

    # ai-review-pending tag was removed
    assert pending_id not in doc["tags"], (
        f"Tag {pending_id} should have been removed; tags={doc['tags']}"
    )

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


@pytest.mark.asyncio
async def test_dry_run_does_not_modify_document(
    paperless_client, dummy_document: int
):
    """
    In dry-run mode, run_batch must return success=1 but leave the document
    completely untouched (tag still present, fields unchanged).
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

    pending_id = paperless_client.get_tag_id(config.tag_pending, create=True)
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
        paperless_client, agent, config, pending_id, custom_field_id, ai_result_field_id
    )

    assert success == 1
    assert failure == 0

    # Fetch again — nothing should have changed
    r_after = paperless_client._client.get(f"/api/documents/{doc_id}/")
    r_after.raise_for_status()
    doc_after = r_after.json()

    assert doc_after["title"] == doc_before["title"], "Dry-run must not change title"
    assert pending_id in doc_after["tags"], "Dry-run must not remove the pending tag"
    cf_map = {cf["field"]: cf["value"] for cf in doc_after.get("custom_fields", [])}
    assert custom_field_id not in cf_map, "Dry-run must not set ai_processed field"
