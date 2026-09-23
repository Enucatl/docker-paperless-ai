"""
Batch runner: orchestrates the Redis-driven document processing pipeline.

Flow per document:
  1. Fetch document metadata from Paperless API
  2. Download original PDF to a temp file
  3. Run SmartDocumentAgent (vision OCR + metadata extraction)
  4. PATCH Paperless (title, date, correspondent, content, custom fields)
  5. SREM doc_id from Redis queue (only on full success)

If any step fails the doc_id remains in the Redis queue and will be retried
on the next run.
"""

import asyncio
import json
import logging
import tempfile
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from paperless_ai.core.config import AgentConfig
from paperless_ai.correspondent import CorrespondentResolver
from paperless_common.paperless import PaperlessClient

if TYPE_CHECKING:
    from paperless_common.queue import TaskQueues

log = logging.getLogger(__name__)

# Set by SIGTERM/SIGINT handler in cli.py; checked between documents.
_shutdown_requested = False


def request_shutdown() -> None:
    global _shutdown_requested
    _shutdown_requested = True


def clear_shutdown_request() -> None:
    global _shutdown_requested
    _shutdown_requested = False


def is_shutdown_requested() -> bool:
    return _shutdown_requested

# Tracks which local server URLs are currently known to be offline.
# Enables log-once-on-down / log-once-on-recovery across poll cycles.
_offline_servers: set[str] = set()
_WEBHOOK_SUPPRESSION_TTL_SECONDS = 300
_model_probe_session = None


async def _check_server_reachable(base_url: str) -> bool:
    """Return True if a local inference server responds to a lightweight probe."""
    for path in ("/health", "/models"):
        try:
            response = await _get_model_probe_session().get(base_url.rstrip("/") + path)
            if response.status_code < 500:
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


def _get_model_probe_session():
    global _model_probe_session
    if _model_probe_session is None:
        import niquests

        _model_probe_session = niquests.AsyncSession(timeout=5.0)
    return _model_probe_session


async def close_model_probe_session() -> None:
    global _model_probe_session
    if _model_probe_session is not None:
        await _model_probe_session.close()
        _model_probe_session = None


async def _suppress_webhook(queue, doc_id: int) -> None:
    await queue.suppress_webhook(doc_id, ttl_seconds=_WEBHOOK_SUPPRESSION_TTL_SECONDS)


async def _clear_webhook_suppression(queue, doc_id: int) -> None:
    await queue.clear_webhook_suppression(doc_id)


async def _record_stage_failure(
    queues: TaskQueues,
    config: AgentConfig,
    *,
    doc_id: int,
    queue_key: str,
    stage_label: str,
) -> None:
    retry_count, moved_to_failed = await queues.mark_failure(
        doc_id,
        queue_key,
        max_attempts=max(1, config.stage_max_attempts),
    )
    if moved_to_failed:
        log.error(
            "Document %d: %s failed %d time(s); moved to failed queue",
            doc_id,
            stage_label,
            retry_count,
        )
    else:
        log.warning(
            "Document %d: %s failed (attempt %d/%d); will retry",
            doc_id,
            stage_label,
            retry_count,
            max(1, config.stage_max_attempts),
        )


async def _run_stage(
    stage_label: str,
    preflight_url: str | None,
    queue_key: str,
    queues: TaskQueues,
    process_fn: Callable[[int], Awaitable[bool | None]],
) -> tuple[int, int]:
    """Shared scaffolding for the OCR and metadata pipeline stages.

    Handles preflight health-check, early-exit on empty queue, gather, and
    success/failure counting.  Stage-specific logic lives in process_fn.
    """
    if preflight_url is not None and not await _check_server_reachable(preflight_url):
        return 0, 0

    pending_ids = await queues.peek_stage(queue_key)
    if not pending_ids:
        return 0, 0

    log.info("%s batch: %d document(s) to process", stage_label, len(pending_ids))
    results = await asyncio.gather(
        *(process_fn(doc_id) for doc_id in sorted(pending_ids))
    )
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
    from paperless_common.queue import TaskQueues

    if await queues.stage_size(TaskQueues.KEY_OCR) == 0:
        return 0, 0

    # Look up tag IDs once for the whole batch
    try:
        tag_ocr_id = await client.get_tag_id(config.tag_ocr, create=False)
    except ValueError:
        log.warning(
            "OCR tag '%s' not found — continuing without tag transition", config.tag_ocr
        )
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
                await _record_stage_failure(
                    queues,
                    config,
                    doc_id=doc_id,
                    queue_key=TaskQueues.KEY_OCR,
                    stage_label="OCR download",
                )
                return False

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                del data

                try:
                    full_text, pages, elapsed = await run_vision_ocr_only(
                        tmp_path, config
                    )
                except Exception as e:
                    log.error("Document %d: OCR failed: %s", doc_id, e)
                    await _record_stage_failure(
                        queues,
                        config,
                        doc_id=doc_id,
                        queue_key=TaskQueues.KEY_OCR,
                        stage_label="OCR",
                    )
                    return False
            finally:
                if tmp_path is not None:
                    Path(tmp_path).unlink(missing_ok=True)

        log.info(
            "Document %d: OCR done — %d pages, %d chars, %.1fs",
            doc_id,
            pages,
            len(full_text),
            elapsed,
        )

        if config.dry_run:
            log.info(
                "Document %d: [dry-run] would write content and transition tag", doc_id
            )
            return True

        # Transition: remove ai:run-ocr, add ai:run-metadata — atomic with content write
        current_tags = [t for t in doc.get("tags", []) if t != tag_ocr_id]
        if tag_metadata_id not in current_tags:
            current_tags.append(tag_metadata_id)

        try:
            await _suppress_webhook(queues, doc_id)
            await client.patch_document(
                doc_id, {"content": full_text, "tags": current_tags}
            )
            log.info(
                "Document %d: content written, transitioned to metadata stage", doc_id
            )
            await queues.remove(doc_id, TaskQueues.KEY_OCR)
            await queues.enqueue_metadata(doc_id)
            return True
        except Exception as e:
            await _clear_webhook_suppression(queues, doc_id)
            log.error("Document %d: PATCH failed: %s", doc_id, e)
            await _record_stage_failure(
                queues,
                config,
                doc_id=doc_id,
                queue_key=TaskQueues.KEY_OCR,
                stage_label="OCR patch",
            )
            return False

    # Preflight: skip the batch when the local OCR server is offline so we
    # don't download PDFs that we can't process yet.
    return await _run_stage(
        "OCR", config.ocr_endpoint or None, TaskQueues.KEY_OCR, queues, _process_one
    )


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
    from paperless_common.queue import TaskQueues

    if await queues.stage_size(TaskQueues.KEY_METADATA) == 0:
        return 0, 0

    strategy = _select_extraction_strategy(config)
    log.info("Metadata batch: using %s", strategy.__class__.__name__)

    try:
        tag_metadata_id = await client.get_tag_id(config.tag_metadata, create=False)
    except ValueError:
        log.warning(
            "Metadata tag '%s' not found — continuing without tag transition",
            config.tag_metadata,
        )
        tag_metadata_id = None

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
            log.warning(
                "Document %d: no content — skipping metadata extraction", doc_id
            )
            await queues.remove(doc_id, TaskQueues.KEY_METADATA)
            return None  # silently skipped — not a success, not a failure

        async with sem:
            try:
                from paperless_ai.agents.smart_graph_agent import (
                    build_metadata_document_context,
                )

                extracted = await strategy.extract(
                    build_metadata_document_context(content), config
                )
            except Exception as e:
                log.error("Document %d: metadata extraction failed: %s", doc_id, e)
                await _record_stage_failure(
                    queues,
                    config,
                    doc_id=doc_id,
                    queue_key=TaskQueues.KEY_METADATA,
                    stage_label="metadata extraction",
                )
                return False

        log.info(
            "Document %d: metadata — title=%r date=%r correspondent=%r",
            doc_id,
            extracted.title,
            extracted.date,
            extracted.correspondent,
        )

        if config.dry_run:
            log.info(
                "Document %d: [dry-run] would write metadata and transition tag", doc_id
            )
            return True

        today = datetime.now(timezone.utc).date().isoformat()
        managed_fields = {custom_field_id, ai_summary_field_id, ai_result_field_id}
        existing_cf = [
            cf
            for cf in doc.get("custom_fields", [])
            if cf["field"] not in managed_fields
        ]

        payload: dict = {
            "custom_fields": existing_cf
            + [
                {"field": custom_field_id, "value": today},
                {
                    "field": ai_summary_field_id,
                    "value": (extracted.summary or "").strip(),
                },
            ],
        }

        if extracted.title:
            payload["title"] = str(extracted.title)[:128]

        if extracted.date:
            if date(1900, 1, 1) <= extracted.date <= date.today():
                payload["created"] = extracted.date.isoformat()
            else:
                log.warning(
                    "Document %d: AI date '%s' out of range, skipping",
                    doc_id,
                    extracted.date,
                )

        correspondent_resolution = None
        if extracted.correspondent and str(extracted.correspondent).strip():
            try:
                observed_name = str(extracted.correspondent).strip()
                resolver = CorrespondentResolver(config.correspondent_match_threshold)
                correspondent_resolution = resolver.resolve(
                    observed_name, await client.get_all_correspondents()
                )
                if correspondent_resolution.action == "new":
                    created = await client.create_correspondent(observed_name)
                    correspondent_resolution = replace(
                        correspondent_resolution,
                        action="created",
                        correspondent_id=int(created["id"]),
                        correspondent_name=str(created["name"]),
                    )
                payload["correspondent"] = correspondent_resolution.correspondent_id
                log.info(
                    "Document %d: correspondent resolution observed=%r action=%s "
                    "candidate=%r id=%s score=%.3f second_best=%s components=%s",
                    doc_id,
                    observed_name,
                    correspondent_resolution.action,
                    correspondent_resolution.correspondent_name,
                    correspondent_resolution.correspondent_id,
                    correspondent_resolution.score,
                    correspondent_resolution.second_best_score,
                    correspondent_resolution.components.to_dict(),
                )
            except Exception as e:
                log.warning("Document %d: correspondent lookup failed: %s", doc_id, e)

        ai_result_json = json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata_model": config.metadata_model,
                "metadata_endpoint": config.metadata_endpoint,
                "paperless_version": client.paperless_version,
                "ai_metadata": {
                    "title": extracted.title,
                    "document_date": extracted.date.isoformat()
                    if extracted.date
                    else None,
                    "correspondent": extracted.correspondent,
                    "summary": extracted.summary,
                },
                **(
                    {
                        "correspondent_resolution": correspondent_resolution.to_audit_dict()
                    }
                    if correspondent_resolution is not None
                    else {}
                ),
            },
            ensure_ascii=False,
        )
        payload["custom_fields"].append(
            {"field": ai_result_field_id, "value": ai_result_json}
        )

        # Remove the metadata tag atomically with the metadata write.
        payload["tags"] = [t for t in doc.get("tags", []) if t != tag_metadata_id]

        try:
            await client.patch_document(doc_id, payload)
            log.info("Document %d: metadata written", doc_id)
            await queues.remove(doc_id, TaskQueues.KEY_METADATA)
            return True
        except Exception as e:
            log.error("Document %d: PATCH failed: %s", doc_id, e)
            await _record_stage_failure(
                queues,
                config,
                doc_id=doc_id,
                queue_key=TaskQueues.KEY_METADATA,
                stage_label="metadata patch",
            )
            return False

    # Preflight: determine which server drives metadata extraction.
    meta_server = config.metadata_endpoint
    return await _run_stage(
        "Metadata", meta_server, TaskQueues.KEY_METADATA, queues, _process_one
    )


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
            except json.JSONDecodeError, TypeError:
                continue
            if "INFERENCE_OCR_MODEL" not in parsed and "ocr_model" not in parsed:
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
