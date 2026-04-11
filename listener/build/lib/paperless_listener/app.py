"""
Thin FastAPI webhook ingress for Paperless document events.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status

from paperless_common.paperless import PaperlessClient
from paperless_common.queue import TaskQueues
from paperless_common.secrets import read_secret

log = logging.getLogger(__name__)

_queues: TaskQueues | None = None
_webhook_secret: str | None = None
_paperless_client: PaperlessClient | None = None
_tag_ocr: str = "ai:run-ocr"
_tag_metadata: str = "ai:run-metadata"
_tag_embed: str = "ai:run-embed"

_DOC_URL_ID_RE = re.compile(r"/documents/(\d+)(?:/|$)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queues, _webhook_secret, _paperless_client
    global _tag_ocr, _tag_metadata, _tag_embed

    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _webhook_secret = read_secret("WEBHOOK_SECRET") or None
    _tag_ocr = os.environ.get("TAG_OCR", os.environ.get("TAG_PENDING", "ai:run-ocr"))
    _tag_metadata = os.environ.get("TAG_METADATA", "ai:run-metadata")
    _tag_embed = os.environ.get("TAG_EMBED", "ai:run-embed")
    paperless_url = os.environ.get("PAPERLESS_URL")
    paperless_token = read_secret("PAPERLESS_TOKEN")

    _queues = TaskQueues(redis_url)
    if paperless_url and paperless_token:
        _paperless_client = PaperlessClient(paperless_url, paperless_token)
        log.info("Ingress Paperless integration enabled (%s)", paperless_url)
    else:
        _paperless_client = None
        log.warning("Ingress Paperless integration disabled")

    log.info(
        "Webhook ingress ready (redis=%s, tags: ocr=%r metadata=%r embed=%r)",
        redis_url,
        _tag_ocr,
        _tag_metadata,
        _tag_embed,
    )
    yield
    if _queues is not None:
        await _queues.close()
    if _paperless_client is not None:
        await _paperless_client.aclose()


app = FastAPI(lifespan=lifespan)


def _extract_doc_id(body: dict) -> int | None:
    doc_url = body.get("doc_url") or body.get("document_url")
    if doc_url:
        match = _DOC_URL_ID_RE.search(str(doc_url))
        if match:
            return int(match.group(1))

    for key in ("document_id", "id"):
        value = body.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _route_to_stage(tags: set[str]) -> str | None:
    if _tag_ocr in tags:
        return TaskQueues.KEY_OCR
    if _tag_metadata in tags:
        return TaskQueues.KEY_METADATA
    if _tag_embed in tags:
        return TaskQueues.KEY_EMBED
    return None


def _parse_tags(body: dict) -> set[str]:
    raw = body.get("tag_list", body.get("document_tags", ""))
    if not raw:
        return set()
    return {tag.strip() for tag in str(raw).split(",") if tag.strip()}


async def _get_current_document_tags(doc_id: int, payload_tags: set[str]) -> set[str]:
    if _paperless_client is None:
        return payload_tags

    try:
        doc = await _paperless_client.get_document(doc_id)
        if doc is None:
            return payload_tags
        tag_ids = doc.get("tags") or []
        if not isinstance(tag_ids, list):
            return payload_tags
        return set(await _paperless_client.get_tag_names(tag_ids))
    except Exception as exc:
        log.warning(
            "Webhook: failed to resolve current tags for document %d: %s", doc_id, exc
        )
        return payload_tags


@app.post("/webhook/document", status_code=202)
async def webhook_document(request: Request) -> Response:
    if _webhook_secret is not None:
        token = request.headers.get("X-Webhook-Token", "")
        if not secrets.compare_digest(token, _webhook_secret):
            log.warning("Webhook: rejected request with invalid token")
            return Response(status_code=status.HTTP_401_UNAUTHORIZED)

    try:
        body = await request.json()
    except Exception:
        log.warning("Webhook received non-JSON body")
        return Response(status_code=400)

    log.info("Webhook payload: %s", body)

    doc_id = _extract_doc_id(body)
    if doc_id is None:
        log.warning("Webhook payload missing document ID: %s", body)
        return Response(status_code=202)

    if _queues is not None:
        tags = await _get_current_document_tags(doc_id, _parse_tags(body))
        stage = _route_to_stage(tags)
        if stage is None:
            added = await _queues.enqueue_refresh(doc_id)
            log.info(
                "Webhook result: document %d → refresh (%s)",
                doc_id,
                "queued" if added else "already pending",
            )
        else:
            added = await _queues.enqueue(doc_id, stage)
            log.info(
                "Webhook result: document %d → %s (%s)",
                doc_id,
                stage.split(":")[-1],
                "queued" if added else "already pending",
            )

    return Response(status_code=202)


@app.get("/health")
async def health() -> dict:
    if _queues is not None:
        pending = await _queues.pending_count()
    else:
        pending = {"ocr": 0, "metadata": 0, "embed": 0, "refresh": 0}
    return {"status": "ok", "pending": pending}
