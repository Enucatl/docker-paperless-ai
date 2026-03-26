"""
Batch runner: orchestrates the Paperless polling loop and document processing.

Responsibilities:
- Poll for pending documents
- Download each document to a temp file (memory-safe)
- Pass the temp file path to the configured agent
- Apply the AgentResult metadata via the Paperless PATCH API
- Clean up temp files in finally blocks
"""

import json
import logging
import os
import tempfile
import time
from datetime import date, datetime, timezone

from agents.base import AgentResult, BaseDocumentAgent
from core.config import AgentConfig
from core.paperless import PaperlessClient

log = logging.getLogger(__name__)

# Set by SIGTERM/SIGINT handler in cli.py; checked between documents.
_shutdown_requested = False


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    return _shutdown_requested


async def process_document(
    doc: dict,
    client: PaperlessClient,
    agent: BaseDocumentAgent,
    config: AgentConfig,
    pending_id: int,
    custom_field_id: int,
    ai_result_field_id: int,
) -> bool:
    """Download, process, and patch a single document. Returns True on success."""
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

        # Run the agent
        try:
            result: AgentResult = await agent.process(tmp_path, existing_hints)
        except ValueError as e:
            log.warning("Document %d: %s — skipping", doc_id, e)
            return False
        except Exception as e:
            log.error("Document %d: agent failed: %s", doc_id, e)
            return False

    finally:
        # Always delete the temp file
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
        log.info("Document %d: [dry-run] would remove tag %d", doc_id, pending_id)
        return True

    log.info("Document %d: PATCHing fields: %s", doc_id, sorted(payload.keys()))
    try:
        client.patch_document(doc_id, payload)
        log.info("Document %d: PATCH OK, removing pending tag", doc_id)
        client.update_tags(doc, remove_id=pending_id, add_id=None)
        log.info("Document %d: done", doc_id)
        return True
    except Exception as e:
        log.error("Document %d: PATCH failed: %s", doc_id, e)
        return False


async def run_batch(
    client: PaperlessClient,
    agent: BaseDocumentAgent,
    config: AgentConfig,
    pending_id: int,
    custom_field_id: int,
    ai_result_field_id: int,
) -> tuple[int, int]:
    """Process all pending documents. Returns (success_count, failure_count)."""
    total = client.count_pending_documents(pending_id)
    if total == 0:
        log.info("No documents tagged '%s'", config.tag_pending)
        return 0, 0

    log.info("Found %d document(s) to process", total)
    success, failure = 0, 0
    for doc in client.iter_pending_documents(pending_id):
        if _shutdown_requested:
            log.info(
                "Shutdown requested — stopping batch after %d/%d documents",
                success + failure,
                total,
            )
            break
        ok = await process_document(
            doc, client, agent, config, pending_id, custom_field_id, ai_result_field_id
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
