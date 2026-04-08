"""
Batch runner: orchestrates the Redis-driven document processing pipeline.

Flow per document:
  1. Fetch document metadata from Paperless API
  2. Download original PDF to a temp file
  3. Run SmartDocumentAgent (vision OCR + metadata extraction)
  4. Chunk OCR text → embed via the embedding API → upsert into Qdrant
  5. PATCH Paperless (title, date, correspondent, content, custom fields)
  6. SREM doc_id from Redis queue (only on full success)

If any step fails the doc_id remains in the Redis queue and will be retried
on the next run.  The embedding step is skipped gracefully when the store or
embedder are not provided (useful for tests and eval mode).
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Optional

from paperless_ai.agents.base import AgentResult, BaseDocumentAgent
from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient

if TYPE_CHECKING:
    from paperless_ai.search.embedder import EmbeddingAPIEmbedder
    from paperless_ai.search.queue import DocumentQueue, TaskQueues
    from paperless_ai.search.qdrant_store import QdrantDocumentStore

log = logging.getLogger(__name__)

# Set by SIGTERM/SIGINT handler in cli.py; checked between documents.
_shutdown_requested = False

# Tracks which local server URLs are currently known to be offline.
# Enables log-once-on-down / log-once-on-recovery across poll cycles.
_offline_servers: set[str] = set()


@dataclass
class SearchMetadata:
    title: Optional[str]
    correspondent: Optional[str]
    document_date: Optional[str]
    summary: Optional[str]
    document_type: Optional[str]
    storage_path: Optional[str]
    tags: list[str]
    year: Optional[str]


def _get_custom_field_value(doc: dict, field_id: int | None) -> Optional[str]:
    """Return a document custom field value by field ID."""
    if field_id is None:
        return None
    for custom_field in doc.get("custom_fields", []):
        if custom_field.get("field") == field_id:
            value = custom_field.get("value")
            return str(value) if value is not None else None
    return None


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    return _shutdown_requested


async def _check_server_reachable(base_url: str) -> bool:
    """Return True if a local model server responds to a lightweight probe.

    Tries GET /health then GET /models (OpenAI-compatible). Logs exactly once
    when a server goes offline and once when it comes back online, suppressing
    repeated warnings between polls so the log stays readable during a long
    GPU-off window.
    """
    import niquests

    for path in ("/health", "/models"):
        try:
            async with niquests.AsyncSession(timeout=5.0) as c:
                r = await c.get(base_url.rstrip("/") + path)
                if r.status_code < 500:
                    if base_url in _offline_servers:
                        log.info("Model server back online: %s", base_url)
                        _offline_servers.discard(base_url)
                    return True
        except Exception:
            continue

    if base_url not in _offline_servers:
        log.warning("Model server unreachable, will retry next poll: %s", base_url)
        _offline_servers.add(base_url)
    return False


async def _embed_and_store(
    doc_id: int,
    full_text: str,
    meta: SearchMetadata,
    config: AgentConfig,
    store: "QdrantDocumentStore",
    embedder: "EmbeddingAPIEmbedder",
) -> None:
    """Chunk text, embed via the embeddings API, and upsert vectors into Qdrant."""
    from paperless_ai.search.chunker import chunk_text
    from paperless_ai.search.qdrant_store import ChunkPayload

    chunks = chunk_text(full_text, config.chunk_size, config.chunk_overlap)
    if not chunks:
        log.info("Document %d: no text to embed, skipping Qdrant upsert", doc_id)
        return

    # Situate chunks before embedding (see hooks.py for strategy selection).
    # The raw chunk is stored in the Qdrant payload so UI snippets show clean text.
    from paperless_ai.core.hooks import situate_chunks
    situated_chunks = await situate_chunks(chunks, full_text, meta, config)

    log.info("Document %d: embedding %d chunk(s) with situated context…", doc_id, len(chunks))
    embeddings = await embedder.embed(situated_chunks)

    # Delete old vectors first so re-processing a document is idempotent
    await store.delete_document(doc_id)

    payloads = [
        ChunkPayload(
            doc_id=doc_id,
            chunk_index=i,
            title=meta.title,
            correspondent=meta.correspondent,
            document_type=meta.document_type,
            storage_path=meta.storage_path,
            tags=meta.tags,
            date=meta.document_date,
            year=meta.year,
            text=chunk,  # raw chunk, not situated, for UI display
        )
        for i, chunk in enumerate(chunks)
    ]
    await store.upsert_chunks(
        payloads,
        dense_vecs=[e.dense for e in embeddings],
        sparse_indices=[e.sparse_indices for e in embeddings],
        sparse_values=[e.sparse_values for e in embeddings],
    )
    log.info("Document %d: upserted %d vector(s) into Qdrant", doc_id, len(chunks))


def _extract_year(date_value: Optional[str]) -> Optional[str]:
    if not date_value:
        return None
    try:
        return str(datetime.fromisoformat(str(date_value)).date().year)
    except ValueError:
        return str(date_value)[:4] if len(str(date_value)) >= 4 else None


async def _build_search_metadata(
    client: PaperlessClient,
    doc: dict,
    *,
    title: Optional[str],
    correspondent: Optional[str],
    document_date: Optional[str],
    summary: Optional[str],
    exclude_tag_ids: set[int] | None = None,
) -> SearchMetadata:
    tag_ids = [
        int(tag_id)
        for tag_id in doc.get("tags", [])
        if not exclude_tag_ids or int(tag_id) not in exclude_tag_ids
    ]
    return SearchMetadata(
        title=title,
        correspondent=correspondent,
        document_date=document_date,
        summary=summary,
        document_type=await client.get_document_type_name(doc.get("document_type")),
        storage_path=await client.get_storage_path_name(doc.get("storage_path")),
        tags=await client.get_tag_names(tag_ids),
        year=_extract_year(document_date),
    )


async def process_document(
    doc: dict,
    client: PaperlessClient,
    agent: BaseDocumentAgent,
    config: AgentConfig,
    custom_field_id: int,
    ai_summary_field_id: int,
    ai_result_field_id: int,
    queue: "DocumentQueue",
    store: QdrantDocumentStore | None = None,
    embedder: "Optional[EmbeddingAPIEmbedder]" = None,
    tag_pending_id: "Optional[int]" = None,
) -> bool:
    """Download, process, embed, and patch a single document. Returns True on success."""
    doc_id = doc["id"]
    log.info("Processing document %d: %s", doc_id, doc.get("title", "(no title)"))

    # Download original file bytes
    try:
        data = await client.download_original(doc_id)
    except Exception as e:
        log.error("Document %d: download failed: %s", doc_id, e)
        return False

    # Write to a named temp file so the agent can open it by path
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        del data  # release the download buffer before heavy processing

        # Build existing metadata hints for the LLM's context
        existing_hints: dict = {}
        if doc.get("title"):
            existing_hints["title"] = doc["title"]
        if doc.get("created"):
            existing_hints["date"] = doc["created"]
        if doc.get("correspondent"):
            correspondent_name = await client.get_correspondent_name(doc["correspondent"])
            if correspondent_name:
                existing_hints["correspondent"] = correspondent_name
        if doc.get("language"):
            existing_hints["language"] = doc["language"]

        # Run the agent (OCR + metadata extraction)
        try:
            result: AgentResult = await agent.process(tmp_path, existing_hints)
        except ValueError as e:
            log.warning("Document %d: %s — skipping", doc_id, e)
            return False
        except Exception as e:
            log.error("Document %d: agent failed: %s", doc_id, e)
            return False

    finally:
        if tmp_path is not None:
            Path(tmp_path).unlink(missing_ok=True)

    meta = result.metadata
    full_text = meta.full_ocr_transcript
    log.info("Document %d: OCR complete — %d chars total", doc_id, len(full_text))
    log.info(
        "Document %d: metadata — title=%r date=%r correspondent=%r",
        doc_id,
        meta.title,
        meta.document_date,
        meta.correspondent,
    )

    # Embed and store vectors (skipped gracefully if store/embedder not configured)
    if store is not None and embedder is not None:
        try:
            search_meta = await _build_search_metadata(
                client,
                doc,
                title=meta.title,
                correspondent=meta.correspondent,
                document_date=meta.document_date,
                summary=meta.summary,
                exclude_tag_ids={tag_pending_id} if tag_pending_id is not None else None,
            )
            await _embed_and_store(doc_id, full_text, search_meta, config, store, embedder)
        except Exception as e:
            log.error("Document %d: embedding failed: %s", doc_id, e)
            return False

    # Build PATCH payload
    today = datetime.now(timezone.utc).date().isoformat()
    managed_fields = {custom_field_id, ai_summary_field_id, ai_result_field_id}
    existing_cf = [
        cf for cf in doc.get("custom_fields", []) if cf["field"] not in managed_fields
    ]
    payload: dict = {
        "content": full_text,
        "custom_fields": existing_cf + [
            {"field": custom_field_id, "value": today},
            {"field": ai_summary_field_id, "value": (meta.summary or "").strip()},
        ],
    }

    if meta.title:
        payload["title"] = str(meta.title)[:128]

    if meta.document_date:
        try:
            parsed = datetime.fromisoformat(str(meta.document_date)).date()
            if date(1900, 1, 1) <= parsed <= date.today():
                payload["created"] = parsed.isoformat()
            else:
                log.warning(
                    "Document %d: AI date '%s' out of range, skipping",
                    doc_id,
                    meta.document_date,
                )
        except ValueError:
            log.warning(
                "Document %d: invalid AI date format '%s', skipping",
                doc_id,
                meta.document_date,
            )

    if meta.correspondent:
        try:
            log.info(
                "Document %d: looking up correspondent '%s'", doc_id, meta.correspondent
            )
            correspondent_id = await client.find_or_create_correspondent(
                str(meta.correspondent).strip()
            )
            payload["correspondent"] = correspondent_id
            log.info("Document %d: correspondent id=%d", doc_id, correspondent_id)
        except Exception as e:
            log.warning("Document %d: correspondent lookup failed: %s", doc_id, e)
    else:
        log.info(
            "Document %d: skipping correspondent (ai=%r, existing=%r)",
            doc_id,
            meta.correspondent,
            doc.get("correspondent"),
        )

    # Remove the pending tag in the same PATCH payload as metadata updates.
    # This ensures atomicity: both tag removal and metadata updates succeed or fail
    # together. This prevents Paperless DOCUMENT_UPDATED webhooks (filtered by the
    # pending tag) from re-queuing the document after AI processing completes.
    # If the PATCH fails for any reason, the document stays in the Redis queue
    # and will be retried on the next batch run.
    if tag_pending_id is not None:
        current_tags = [t for t in doc.get("tags", []) if t != tag_pending_id]
        payload["tags"] = current_tags
        log.info(
            "Document %d: removing pending tag (id=%d) from tags",
            doc_id, tag_pending_id,
        )

    ai_result_json = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_s": result.elapsed_s,
            "ocr_method": result.ocr_method,
            "ocr_model": config.ocr_model,
            "ocr_api_base": config.ocr_api_base,
            "metadata_model": config.metadata_model,
            "metadata_api_base": config.metadata_api_base,
            "pages": result.pages,
            "chars": result.chars,
            "paperless_version": client.paperless_version,
            "ai_metadata": {
                "title": meta.title,
                "document_date": meta.document_date,
                "correspondent": meta.correspondent,
                "summary": meta.summary,
            },
        },
        ensure_ascii=False,
    )
    payload["custom_fields"].append({"field": ai_result_field_id, "value": ai_result_json})

    if config.dry_run:
        log.info(
            "Document %d: [dry-run] would PATCH fields: %s",
            doc_id,
            sorted(payload.keys()),
        )
        log.info("Document %d: [dry-run] would remove from Redis queue", doc_id)
        return True

    log.info("Document %d: PATCHing fields: %s", doc_id, sorted(payload.keys()))
    try:
        await client.patch_document(doc_id, payload)
        log.info("Document %d: PATCH OK, removing from queue", doc_id)
        await queue.remove(doc_id)
        log.info("Document %d: done", doc_id)
        return True
    except Exception as e:
        log.error("Document %d: PATCH failed: %s", doc_id, e)
        return False


async def run_batch(
    client: PaperlessClient,
    agent: BaseDocumentAgent,
    config: AgentConfig,
    custom_field_id: int,
    ai_summary_field_id: int,
    ai_result_field_id: int,
    queue: "DocumentQueue",
    store: QdrantDocumentStore | None = None,
    embedder: "Optional[EmbeddingAPIEmbedder]" = None,
) -> tuple[int, int]:
    """Process all documents in the Redis queue concurrently. Returns (success_count, failure_count)."""
    pending_ids = await queue.peek_all()
    if not pending_ids:
        log.info("No documents pending in queue")
        return 0, 0

    log.info(
        "Found %d document(s) to process (concurrency=%d)",
        len(pending_ids),
        config.ocr_concurrency,
    )

    # Look up the pending tag ID once so process_document can remove it
    # atomically with the metadata PATCH, preventing DOCUMENT_UPDATED
    # webhook loops when the Paperless workflow filters by this tag.
    tag_pending_id: Optional[int] = None
    try:
        tag_pending_id = await client.get_tag_id(config.tag_pending, create=False)
    except ValueError:
        log.debug(
            "Pending tag '%s' not found — will not remove it on processing",
            config.tag_pending,
        )

    sem = asyncio.Semaphore(config.ocr_concurrency)

    async def _process_one(doc_id: int) -> bool:
        if _shutdown_requested:
            return False
        doc = await client.get_document(doc_id)
        if doc is None:
            log.warning("Document %d not found in Paperless — removing from queue", doc_id)
            await queue.remove(doc_id)
            return None  # silently removed — not a success, not a failure
        async with sem:
            return await process_document(
                doc, client, agent, config, custom_field_id, ai_summary_field_id, ai_result_field_id,
                queue, store, embedder, tag_pending_id=tag_pending_id,
            )

    results = await asyncio.gather(*(_process_one(doc_id) for doc_id in sorted(pending_ids)))

    if _shutdown_requested:
        log.info("Shutdown requested — batch may be incomplete")

    success = sum(1 for ok in results if ok is True)
    failure = sum(1 for ok in results if ok is False)
    return success, failure


async def _run_stage(
    stage_label: str,
    preflight_url: str | None,
    queue_key: str,
    queues: TaskQueues,
    process_fn: Callable[[int], Awaitable[bool | None]],
) -> tuple[int, int]:
    """Shared scaffolding for the three pipeline stages (OCR / metadata / embed).

    Handles preflight health-check, early-exit on empty queue, gather, and
    success/failure counting.  Stage-specific logic lives in process_fn.
    """
    if preflight_url is not None and not await _check_server_reachable(preflight_url):
        return 0, 0

    pending_ids = await queues.peek_stage(queue_key)
    if not pending_ids:
        return 0, 0

    log.info("%s batch: %d document(s) to process", stage_label, len(pending_ids))
    results = await asyncio.gather(*(process_fn(doc_id) for doc_id in sorted(pending_ids)))
    success = sum(1 for ok in results if ok is True)
    failure = sum(1 for ok in results if ok is False)
    return success, failure


async def run_ocr_batch(
    client: PaperlessClient,
    config: AgentConfig,
    queues: TaskQueues,
) -> tuple[int, int]:
    """OCR stage: download PDF, run vision OCR, write content, transition tag to ai:run-metadata.

    Directly enqueues processed docs to the metadata Redis queue so the pipeline
    advances without relying on webhook timing.
    """
    from paperless_ai.agents.smart_graph_agent import run_vision_ocr_only
    from paperless_ai.search.queue import TaskQueues

    # Look up tag IDs once for the whole batch
    try:
        tag_ocr_id = await client.get_tag_id(config.tag_ocr, create=False)
    except ValueError:
        log.warning("OCR tag '%s' not found — continuing without tag transition", config.tag_ocr)
        tag_ocr_id = None

    try:
        tag_metadata_id = await client.get_tag_id(config.tag_metadata, create=True)
    except Exception as e:
        log.error("Cannot resolve metadata tag '%s': %s", config.tag_metadata, e)
        pending_ids = await queues.peek_stage(TaskQueues.KEY_OCR)
        return 0, len(pending_ids)

    sem = asyncio.Semaphore(config.ocr_concurrency)

    async def _process_one(doc_id: int) -> bool | None:
        if _shutdown_requested:
            return False
        doc = await client.get_document(doc_id)
        if doc is None:
            log.warning("Document %d not found — removing from OCR queue", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_OCR)
            return None  # silently removed — not a success, not a failure

        async with sem:
            try:
                data = await client.download_original(doc_id)
            except Exception as e:
                log.error("Document %d: download failed: %s", doc_id, e)
                return False

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                del data

                try:
                    full_text, pages, elapsed = await run_vision_ocr_only(tmp_path, config)
                except Exception as e:
                    log.error("Document %d: OCR failed: %s", doc_id, e)
                    return False
            finally:
                if tmp_path is not None:
                    Path(tmp_path).unlink(missing_ok=True)

        log.info(
            "Document %d: OCR done — %d pages, %d chars, %.1fs",
            doc_id, pages, len(full_text), elapsed,
        )

        if config.dry_run:
            log.info("Document %d: [dry-run] would write content and transition tag", doc_id)
            return True

        # Transition: remove ai:run-ocr, add ai:run-metadata — atomic with content write
        current_tags = [t for t in doc.get("tags", []) if t != tag_ocr_id]
        if tag_metadata_id not in current_tags:
            current_tags.append(tag_metadata_id)

        try:
            await client.patch_document(doc_id, {"content": full_text, "tags": current_tags})
            log.info("Document %d: content written, transitioned to metadata stage", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_OCR)
            await queues.enqueue_metadata(doc_id)
            return True
        except Exception as e:
            log.error("Document %d: PATCH failed: %s", doc_id, e)
            return False

    # Preflight: skip the batch when the local OCR server is offline so we
    # don't download PDFs that we can't process yet.
    return await _run_stage("OCR", config.ocr_api_base or None, TaskQueues.KEY_OCR, queues, _process_one)


async def run_metadata_batch(
    client: PaperlessClient,
    config: AgentConfig,
    queues: TaskQueues,
    custom_field_id: int,
    ai_summary_field_id: int,
    ai_result_field_id: int,
) -> tuple[int, int]:
    """Metadata stage: read content from Paperless, run LLM, write metadata, transition tag.

    No PDF download. Reads the content written by the OCR stage.
    """
    from paperless_ai.agents.smart_graph_agent import _select_extraction_strategy
    from paperless_ai.search.queue import TaskQueues

    strategy = _select_extraction_strategy(config)
    log.info("Metadata batch: using %s", strategy.__class__.__name__)

    try:
        tag_metadata_id = await client.get_tag_id(config.tag_metadata, create=False)
    except ValueError:
        log.warning("Metadata tag '%s' not found — continuing without tag transition", config.tag_metadata)
        tag_metadata_id = None

    try:
        tag_embed_id = await client.get_tag_id(config.tag_embed, create=True)
    except Exception as e:
        log.error("Cannot resolve embed tag '%s': %s", config.tag_embed, e)
        pending_ids = await queues.peek_stage(TaskQueues.KEY_METADATA)
        return 0, len(pending_ids)

    sem = asyncio.Semaphore(config.ocr_concurrency)

    async def _process_one(doc_id: int) -> bool | None:
        if _shutdown_requested:
            return False
        doc = await client.get_document_with_content(doc_id)
        if doc is None:
            log.warning("Document %d not found — removing from metadata queue", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_METADATA)
            return None  # silently removed — not a success, not a failure

        content = doc.get("content") or ""
        if not content.strip():
            log.warning("Document %d: no content — skipping metadata extraction", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_METADATA)
            return None  # silently skipped — not a success, not a failure

        async with sem:
            # Truncate to first 4000 + last 2000 chars (same as SmartDocumentAgent)
            if len(content) > 6000:
                snippet = content[:4000] + "\n...\n" + content[-2000:]
            else:
                snippet = content

            try:
                extracted = await strategy.extract(snippet, config)
            except Exception as e:
                log.error("Document %d: metadata extraction failed: %s", doc_id, e)
                return False

        log.info(
            "Document %d: metadata — title=%r date=%r correspondent=%r",
            doc_id, extracted.title, extracted.date, extracted.correspondent,
        )

        if config.dry_run:
            log.info("Document %d: [dry-run] would write metadata and transition tag", doc_id)
            return True

        today = datetime.now(timezone.utc).date().isoformat()
        managed_fields = {custom_field_id, ai_summary_field_id, ai_result_field_id}
        existing_cf = [cf for cf in doc.get("custom_fields", []) if cf["field"] not in managed_fields]

        payload: dict = {
            "custom_fields": existing_cf + [
                {"field": custom_field_id, "value": today},
                {"field": ai_summary_field_id, "value": (extracted.summary or "").strip()},
            ],
        }

        if extracted.title:
            payload["title"] = str(extracted.title)[:128]

        if extracted.date:
            if date(1900, 1, 1) <= extracted.date <= date.today():
                payload["created"] = extracted.date.isoformat()
            else:
                log.warning("Document %d: AI date '%s' out of range, skipping", doc_id, extracted.date)

        if extracted.correspondent:
            try:
                correspondent_id = await client.find_or_create_correspondent(
                    str(extracted.correspondent).strip()
                )
                payload["correspondent"] = correspondent_id
            except Exception as e:
                log.warning("Document %d: correspondent lookup failed: %s", doc_id, e)

        ai_result_json = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata_model": config.metadata_model,
                "metadata_api_base": config.metadata_api_base,
                "paperless_version": client.paperless_version,
                "ai_metadata": {
                    "title": extracted.title,
                    "document_date": extracted.date.isoformat() if extracted.date else None,
                    "correspondent": extracted.correspondent,
                    "summary": extracted.summary,
                },
            },
            ensure_ascii=False,
        )
        payload["custom_fields"].append({"field": ai_result_field_id, "value": ai_result_json})

        # Transition: remove ai:run-metadata, add ai:run-embed — atomic with metadata write
        current_tags = [t for t in doc.get("tags", []) if t != tag_metadata_id]
        if tag_embed_id not in current_tags:
            current_tags.append(tag_embed_id)
        payload["tags"] = current_tags

        try:
            await client.patch_document(doc_id, payload)
            log.info("Document %d: metadata written, transitioned to embed stage", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_METADATA)
            await queues.enqueue_embed(doc_id)
            return True
        except Exception as e:
            log.error("Document %d: PATCH failed: %s", doc_id, e)
            return False

    # Preflight: determine which server drives metadata extraction.
    meta_server = config.metadata_api_base
    return await _run_stage("Metadata", meta_server, TaskQueues.KEY_METADATA, queues, _process_one)


async def run_embed_batch(
    client: PaperlessClient,
    config: AgentConfig,
    queues: TaskQueues,
    store: QdrantDocumentStore | None = None,
    embedder: Optional[EmbeddingAPIEmbedder] = None,
) -> tuple[int, int]:
    """Embed stage: read content + metadata from Paperless, embed, upsert Qdrant, remove tag.

    Zero LLM calls. Can be used to rebuild the index by pushing any doc IDs to queue:embed.
    """
    from paperless_ai.search.queue import TaskQueues

    try:
        tag_embed_id = await client.get_tag_id(config.tag_embed, create=False)
    except ValueError:
        log.warning("Embed tag '%s' not found — will not remove it", config.tag_embed)
        tag_embed_id = None

    try:
        ai_summary_field_id = await client.get_or_create_custom_field(
            "ai_summary", data_type="longtext"
        )
    except Exception as e:
        log.warning("Could not resolve ai_summary custom field for embed stage: %s", e)
        ai_summary_field_id = None

    sem = asyncio.Semaphore(config.ocr_concurrency)

    async def _process_one(doc_id: int) -> bool | None:
        if _shutdown_requested:
            return False
        doc = await client.get_document_with_content(doc_id)
        if doc is None:
            log.warning("Document %d not found — removing from embed queue", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_EMBED)
            return None  # silently removed — not a success, not a failure

        content = doc.get("content") or ""

        async with sem:
            if store is not None and embedder is not None and content.strip():
                try:
                    search_meta = await _build_search_metadata(
                        client,
                        doc,
                        title=doc.get("title"),
                        correspondent=await client.get_correspondent_name(doc["correspondent"])
                        if doc.get("correspondent")
                        else None,
                        document_date=doc.get("created"),
                        summary=_get_custom_field_value(doc, ai_summary_field_id),
                        exclude_tag_ids={tag_embed_id} if tag_embed_id is not None else None,
                    )
                    await _embed_and_store(doc_id, content, search_meta, config, store, embedder)
                except Exception as e:
                    log.error("Document %d: embedding failed: %s", doc_id, e)
                    return False
            elif not content.strip():
                log.info("Document %d: no content — skipping embedding", doc_id)

        if config.dry_run:
            log.info("Document %d: [dry-run] would remove embed tag", doc_id)
            return True

        # Remove the embed tag
        if tag_embed_id is not None:
            current_tags = [t for t in doc.get("tags", []) if t != tag_embed_id]
            try:
                await client.patch_document(doc_id, {"tags": current_tags})
                log.info("Document %d: embedded, removed embed tag", doc_id)
            except Exception as e:
                log.error("Document %d: tag removal PATCH failed: %s", doc_id, e)
                return False

        await queues.remove(doc_id, TaskQueues.KEY_EMBED)
        return True

    # Preflight: the embedding API server is optional — bail if it is offline so
    # we don't leave documents stuck in the embed queue.
    preflight = config.embedding_api_base if embedder is not None else None
    return await _run_stage("Embed", preflight, TaskQueues.KEY_EMBED, queues, _process_one)


async def purge_ai_notes(client: PaperlessClient, dry_run: bool) -> None:
    """Delete all notes that were written by previous AI processing runs."""
    docs = await client.iter_all_documents()
    log.info("Scanning %d document(s) for AI-generated notes", len(docs))
    deleted = 0
    for doc in docs:
        doc_id = doc["id"]
        try:
            notes = await client.list_notes(doc_id)
        except Exception as e:
            log.warning("Document %d: could not fetch notes: %s", doc_id, e)
            continue
        for note in notes:
            text = note.get("note", "")
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if "OCR_MODEL" not in parsed and "ocr_model" not in parsed:
                continue
            note_id = note["id"]
            if dry_run:
                log.info("Document %d: [dry-run] would delete note %d", doc_id, note_id)
            else:
                try:
                    await client.delete_note(doc_id, note_id)
                    log.info("Document %d: deleted note %d", doc_id, note_id)
                    deleted += 1
                except Exception as e:
                    log.warning(
                        "Document %d: could not delete note %d: %s", doc_id, note_id, e
                    )
    log.info("Done. %d note(s) deleted.", deleted)
