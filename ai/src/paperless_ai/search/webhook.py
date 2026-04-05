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

import asyncio
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import NamedVector

from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.queue import TaskQueues
from paperless_ai.search.qdrant_store import COLLECTION

log = logging.getLogger(__name__)

_queues: TaskQueues | None = None
_webhook_secret: str | None = None
_lazy_embedder: LocalLazySearchEmbedder | None = None
_idle_task: asyncio.Task | None = None
_tag_ocr: str = "ai:run-ocr"
_tag_metadata: str = "ai:run-metadata"
_tag_embed: str = "ai:run-embed"

# Matches the numeric document ID anywhere in a Paperless document URL.
# e.g. "https://paperless.home/documents/42/detail" → "42"
_DOC_URL_ID_RE = re.compile(r"/documents/(\d+)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queues, _webhook_secret, _tag_ocr, _tag_metadata, _tag_embed
    global _lazy_embedder, _idle_task
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
    _lazy_embedder = LocalLazySearchEmbedder()
    _idle_task = asyncio.create_task(_lazy_embedder.idle_watcher())
    log.info("Webhook listener ready (redis=%s, tags: ocr=%r metadata=%r embed=%r)",
             redis_url, _tag_ocr, _tag_metadata, _tag_embed)
    yield
    _idle_task.cancel()
    with suppress(asyncio.CancelledError):
        await _idle_task
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


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> list[int]:
    """Search indexed documents and return matching Paperless doc_ids.

    Uses the local CPU FastEmbed model so the endpoint works even when the
    GPU Infinity server is powered off.
    """
    if _lazy_embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not ready")

    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")

    result = await _lazy_embedder.embed_query(q)

    qdrant = AsyncQdrantClient(url=qdrant_url)
    try:
        hits = await qdrant.search(
            collection_name=COLLECTION,
            query_vector=NamedVector(name="dense", vector=result.dense),
            limit=limit,
            with_payload=True,
        )
    finally:
        await qdrant.close()

    # Deduplicate doc_ids while preserving score order.
    seen: set[int] = set()
    doc_ids: list[int] = []
    for hit in hits:
        doc_id = hit.payload.get("doc_id") if hit.payload else None
        if doc_id is not None and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


@app.get("/health")
async def health() -> dict:
    pending = await _queues.pending_count() if _queues else {"ocr": 0, "metadata": 0, "embed": 0}
    return {"status": "ok", "pending": pending}
