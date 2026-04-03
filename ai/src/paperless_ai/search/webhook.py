"""
FastAPI webhook listener for Paperless-ngx document events.

Paperless-ngx sends a POST request whose body is fully user-configured via
key-value pairs with Jinja2 placeholders.  Configure in Paperless:

  Settings → Workflows
    Trigger:  Document Added / Document Updated
    Action:   Webhook
      URL:    http://webhook-listener:8001/webhook/document
      Body (JSON, key-value):
        doc_url          →  {{doc_url}}
        document_tags    →  {{document_tags}}
      Headers:
        X-Webhook-Token: <WEBHOOK_SECRET value>

Routing (tag-driven):
  ai:run-ocr      → queue:ocr      (vision OCR stage)
  ai:run-metadata → queue:metadata (LLM metadata extraction stage)
  ai:run-embed    → queue:embed    (embedding stage)
  (no ai:run-*)   → queue:embed    (human edit — keep index in sync)

Authentication is optional: if WEBHOOK_SECRET is set, the endpoint validates
the X-Webhook-Token header using constant-time comparison.

Health endpoint:
    GET /health → {"status": "ok", "pending": {"ocr": N, "metadata": N, "embed": N}}
"""

import logging
import os
import re
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status

from paperless_ai.search.queue import TaskQueues

log = logging.getLogger(__name__)

_queues: TaskQueues | None = None
_webhook_secret: str | None = None
_tag_ocr: str = "ai:run-ocr"
_tag_metadata: str = "ai:run-metadata"
_tag_embed: str = "ai:run-embed"

# Matches the numeric document ID anywhere in a Paperless document URL.
# e.g. "https://paperless.home/documents/42/detail" → "42"
_DOC_URL_ID_RE = re.compile(r"/documents/(\d+)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queues, _webhook_secret, _tag_ocr, _tag_metadata, _tag_embed
    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _webhook_secret = os.environ.get("WEBHOOK_SECRET") or None
    _tag_ocr = os.environ.get("TAG_OCR", os.environ.get("TAG_PENDING", "ai:run-ocr"))
    _tag_metadata = os.environ.get("TAG_METADATA", "ai:run-metadata")
    _tag_embed = os.environ.get("TAG_EMBED", "ai:run-embed")

    if _webhook_secret:
        log.info("Webhook authentication enabled")
    else:
        log.warning("WEBHOOK_SECRET not set — webhook endpoint is unauthenticated")
    _queues = TaskQueues(redis_url)
    log.info("Webhook listener ready (redis=%s, tags: ocr=%r metadata=%r embed=%r)",
             redis_url, _tag_ocr, _tag_metadata, _tag_embed)
    yield
    if _queues:
        await _queues.close()


app = FastAPI(lifespan=lifespan)


def _extract_doc_id(body: dict) -> int | None:
    """Extract the document ID from a Paperless webhook payload.

    Tries, in order:
    1. "doc_url" key  — extract numeric ID from the URL path (recommended setup)
    2. "document_id" key — plain integer, for custom webhook bodies
    3. "id" key — flat fallback
    """
    doc_url = body.get("doc_url") or body.get("document_url")
    if doc_url:
        m = _DOC_URL_ID_RE.search(str(doc_url))
        if m:
            return int(m.group(1))

    for key in ("document_id", "id"):
        val = body.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    return None


def _route_to_stage(tags: set[str]) -> str:
    """Determine which queue stage to route to based on document tags.

    Priority: ocr > metadata > embed > embed (fallback for human edits).
    """
    if _tag_ocr in tags:
        return TaskQueues.KEY_OCR
    if _tag_metadata in tags:
        return TaskQueues.KEY_METADATA
    # ai:run-embed tag OR no ai:run-* tag at all → (re-)embed
    return TaskQueues.KEY_EMBED


def _parse_tags(body: dict) -> set[str]:
    """Parse document_tags from the webhook payload.

    Paperless sends {{document_tags}} as a comma-separated string of tag names.
    """
    raw = body.get("document_tags", "")
    if not raw:
        return set()
    return {t.strip() for t in str(raw).split(",") if t.strip()}


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

    doc_id = _extract_doc_id(body)
    if doc_id is None:
        log.warning("Webhook payload missing document ID: %s", body)
        return Response(status_code=202)  # Accept anyway — don't make Paperless retry

    if _queues:
        tags = _parse_tags(body)
        stage = _route_to_stage(tags)
        added = await _queues.enqueue(doc_id, stage)
        log.info(
            "Webhook: document %d → %s (%s)",
            doc_id,
            stage.split(":")[-1],
            "queued" if added else "already pending",
        )

    return Response(status_code=202)


@app.get("/health")
async def health() -> dict:
    pending = await _queues.pending_count() if _queues else {"ocr": 0, "metadata": 0, "embed": 0}
    return {"status": "ok", "pending": pending}
