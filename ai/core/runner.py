"""
Batch runner: orchestrates the Redis-driven document processing pipeline.

Flow per document:
  1. Fetch document metadata from Paperless API
  2. Download original PDF to a temp file
  3. Run SmartDocumentAgent (vision OCR + metadata extraction)
  4. Chunk OCR text → embed via Infinity → upsert into Qdrant
  5. PATCH Paperless (title, date, correspondent, content, custom fields)
  6. SREM doc_id from Redis queue (only on full success)

If any step fails the doc_id remains in the Redis queue and will be retried
on the next run.  The embedding step is skipped gracefully when the store or
embedder are not provided (useful for tests and eval mode).
"""

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Optional

from agents.base import AgentResult, BaseDocumentAgent
from core.config import AgentConfig
from core.paperless import PaperlessClient

if TYPE_CHECKING:
    from search.embedder import InfinityEmbedder
    from search.queue import DocumentQueue
    from search.qdrant_store import QdrantDocumentStore

log = logging.getLogger(__name__)

# Set by SIGTERM/SIGINT handler in cli.py; checked between documents.
_shutdown_requested = False


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    return _shutdown_requested


async def _embed_and_store(
    doc_id: int,
    full_text: str,
    meta,
    config: AgentConfig,
    store: "QdrantDocumentStore",
    embedder: "InfinityEmbedder",
) -> None:
    """Chunk text, embed via Infinity, and upsert vectors into Qdrant."""
    from search.chunker import chunk_text
    from search.qdrant_store import ChunkPayload

    chunks = chunk_text(full_text, config.chunk_max_chars, config.chunk_overlap)
    if not chunks:
        log.info("Document %d: no text to embed, skipping Qdrant upsert", doc_id)
        return

    log.info("Document %d: embedding %d chunk(s)…", doc_id, len(chunks))
    embeddings = await embedder.embed(chunks)

    # Delete old vectors first so re-processing a document is idempotent
    await store.delete_document(doc_id)

    payloads = [
        ChunkPayload(
            doc_id=doc_id,
            chunk_index=i,
            title=meta.title,
            correspondent=meta.correspondent,
            date=meta.document_date,
            text=chunk,
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


async def process_document(
    doc: dict,
    client: PaperlessClient,
    agent: BaseDocumentAgent,
    config: AgentConfig,
    custom_field_id: int,
    ai_result_field_id: int,
    queue: "DocumentQueue",
    store: "Optional[QdrantDocumentStore]" = None,
    embedder: "Optional[InfinityEmbedder]" = None,
) -> bool:
    """Download, process, embed, and patch a single document. Returns True on success."""
    doc_id = doc["id"]
    log.info("Processing document %d: %s", doc_id, doc.get("title", "(no title)"))

    # Download original file bytes
    try:
        data = client.download_original(doc_id)
    except Exception as e:
        log.error("Document %d: download failed: %s", doc_id, e)
        return False

    # Write to a named temp file so the agent can open it by path
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        os.write(tmp_fd, data)
        os.close(tmp_fd)
        tmp_fd = -1  # mark as closed
        del data  # release the download buffer before heavy processing

        # Build existing metadata hints for the LLM's context
        existing_hints: dict = {}
        if doc.get("title"):
            existing_hints["title"] = doc["title"]
        if doc.get("created_date"):
            existing_hints["date"] = doc["created_date"]
        if doc.get("correspondent"):
            correspondent_name = client.get_correspondent_name(doc["correspondent"])
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
        if tmp_fd != -1:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

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
            await _embed_and_store(doc_id, full_text, meta, config, store, embedder)
        except Exception as e:
            log.error("Document %d: embedding failed: %s", doc_id, e)
            return False

    # Build PATCH payload
    today = datetime.now(timezone.utc).date().isoformat()
    managed_fields = {custom_field_id, ai_result_field_id}
    existing_cf = [
        cf for cf in doc.get("custom_fields", []) if cf["field"] not in managed_fields
    ]
    payload: dict = {
        "content": full_text,
        "custom_fields": existing_cf + [{"field": custom_field_id, "value": today}],
    }

    if meta.title:
        payload["title"] = str(meta.title)[:128]

    if meta.document_date:
        try:
            parsed = datetime.fromisoformat(str(meta.document_date)).date()
            if date(1900, 1, 1) <= parsed <= date.today():
                payload["created_date"] = parsed.isoformat()
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
            correspondent_id = client.find_or_create_correspondent(
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

    ai_result_json = json.dumps(
        {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "elapsed_s": result.elapsed_s,
            "ocr_method": result.ocr_method,
            "ocr_model": config.ocr_model,
            "ocr_api_base": config.ocr_api_base,
            "metadata_model": config.effective_metadata_model,
            "metadata_api_base": config.metadata_api_base,
            "pages": result.pages,
            "chars": result.chars,
            "paperless_version": client.paperless_version,
            "ai_metadata": {
                "title": meta.title,
                "document_date": meta.document_date,
                "correspondent": meta.correspondent,
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
        client.patch_document(doc_id, payload)
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
    ai_result_field_id: int,
    queue: "DocumentQueue",
    store: "Optional[QdrantDocumentStore]" = None,
    embedder: "Optional[InfinityEmbedder]" = None,
) -> tuple[int, int]:
    """Process all documents in the Redis queue. Returns (success_count, failure_count)."""
    pending_ids = await queue.peek_all()
    if not pending_ids:
        log.info("No documents pending in queue")
        return 0, 0

    log.info("Found %d document(s) to process", len(pending_ids))
    success, failure = 0, 0

    for doc_id in sorted(pending_ids):
        if _shutdown_requested:
            log.info(
                "Shutdown requested — stopping batch after %d/%d documents",
                success + failure,
                len(pending_ids),
            )
            break

        doc = client.get_document(doc_id)
        if doc is None:
            log.warning("Document %d not found in Paperless — removing from queue", doc_id)
            await queue.remove(doc_id)
            continue

        ok = await process_document(
            doc, client, agent, config, custom_field_id, ai_result_field_id,
            queue, store, embedder,
        )
        if ok:
            success += 1
        else:
            failure += 1

    return success, failure


def purge_ai_notes(client: PaperlessClient, dry_run: bool) -> None:
    """Delete all notes that were written by previous AI processing runs."""
    docs = client.iter_all_documents()
    log.info("Scanning %d document(s) for AI-generated notes", len(docs))
    deleted = 0
    for doc in docs:
        doc_id = doc["id"]
        try:
            notes = client.list_notes(doc_id)
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
                    client.delete_note(doc_id, note_id)
                    log.info("Document %d: deleted note %d", doc_id, note_id)
                    deleted += 1
                except Exception as e:
                    log.warning(
                        "Document %d: could not delete note %d: %s", doc_id, note_id, e
                    )
    log.info("Done. %d note(s) deleted.", deleted)
