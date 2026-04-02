"""
FastAPI webhook listener for Paperless-ngx document events.

Paperless-ngx sends a POST request whose body is fully user-configured via
key-value pairs with Jinja2 placeholders.  There is no automatic {{doc_id}}
placeholder, so the recommended setup is to pass the document URL and extract
the numeric ID from it.

Configure in Paperless: Settings → Workflows
  Trigger:  Document Added  (and/or Document Updated)
  Action:   Webhook
    URL:    http://webhook-listener:8001/webhook/document
    Body (JSON, key-value):
      doc_url  →  {{doc_url}}

The listener extracts the document ID from the URL path
(e.g. "https://paperless.home/documents/42/detail" → 42).
It also accepts a plain "document_id" key as a fallback for custom setups.

Health endpoint:
    GET /health → {"status": "ok", "pending": <count>}
"""

import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from search.queue import DocumentQueue

log = logging.getLogger(__name__)

_queue: DocumentQueue | None = None

# Matches the numeric document ID anywhere in a Paperless document URL.
# e.g. "https://paperless.home/documents/42/detail" → "42"
_DOC_URL_ID_RE = re.compile(r"/documents/(\d+)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queue
    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _queue = DocumentQueue(redis_url)
    log.info("Webhook listener ready (redis=%s)", redis_url)
    yield
    if _queue:
        await _queue.close()


app = FastAPI(lifespan=lifespan)


def _extract_doc_id(body: dict) -> int | None:
    """Extract the document ID from a Paperless webhook payload.

    Tries, in order:
    1. "doc_url" key  — extract numeric ID from the URL path (recommended setup)
    2. "document_id" key — plain integer, for custom webhook bodies
    3. "id" key — flat fallback
    """
    # Primary: parse from the document URL that Paperless provides via {{doc_url}}
    doc_url = body.get("doc_url") or body.get("document_url")
    if doc_url:
        m = _DOC_URL_ID_RE.search(str(doc_url))
        if m:
            return int(m.group(1))

    # Fallback: plain integer fields
    for key in ("document_id", "id"):
        val = body.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass

    return None


@app.post("/webhook/document", status_code=202)
async def webhook_document(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        log.warning("Webhook received non-JSON body")
        return Response(status_code=400)

    doc_id = _extract_doc_id(body)
    if doc_id is None:
        log.warning("Webhook payload missing document ID: %s", body)
        return Response(status_code=202)  # Accept anyway — don't make Paperless retry

    if _queue:
        added = await _queue.enqueue(doc_id)
        log.info("Webhook: document %d %s", doc_id, "queued" if added else "already pending")

    return Response(status_code=202)


@app.get("/health")
async def health() -> dict:
    pending = await _queue.pending_count() if _queue else 0
    return {"status": "ok", "pending": pending}
