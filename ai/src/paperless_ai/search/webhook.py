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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.queue import TaskQueues
from paperless_ai.search.retriever import (
    ScoredDoc,
    dense_search,
    keyword_search,
    llm_rerank,
    rrf_fuse,
)

log = logging.getLogger(__name__)

# Task queue and webhooks
_queues: TaskQueues | None = None
_webhook_secret: str | None = None
_lazy_embedder: LocalLazySearchEmbedder | None = None
_idle_task: asyncio.Task | None = None
_tag_ocr: str = "ai:run-ocr"
_tag_metadata: str = "ai:run-metadata"
_tag_embed: str = "ai:run-embed"

# Hybrid search
_paperless_client: PaperlessClient | None = None
_qdrant_url: str = "http://qdrant:6333"
_rerank_model: str | None = None
_rerank_api_base: str | None = None

# Retrieval hyperparameters
K = 25  # max chunks from dense search
N = 50  # max candidates for RRF before LLM reranking
RRF_K = 60  # RRF smoothing constant

# Matches the numeric document ID anywhere in a Paperless document URL.
# e.g. "https://paperless.home/documents/42/detail" → "42"
_DOC_URL_ID_RE = re.compile(r"/documents/(\d+)(?:/|$)")


def _read_secret(env_var: str) -> str | None:
    """Read env var, or if FOO_FILE is set, read its content from that file.

    Gracefully handles missing or inaccessible secret files.
    """
    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        p = Path(file_path)
        try:
            if p.is_file():
                return p.read_text().strip()
        except (OSError, ValueError):
            pass
    return os.environ.get(env_var)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queues, _webhook_secret, _tag_ocr, _tag_metadata, _tag_embed
    global _lazy_embedder, _idle_task, _paperless_client, _qdrant_url
    global _rerank_model, _rerank_api_base

    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _webhook_secret = os.environ.get("WEBHOOK_SECRET") or None
    _tag_ocr = os.environ.get("TAG_OCR", os.environ.get("TAG_PENDING", "ai:run-ocr"))
    _tag_metadata = os.environ.get("TAG_METADATA", "ai:run-metadata")
    _tag_embed = os.environ.get("TAG_EMBED", "ai:run-embed")

    # Hybrid search configuration
    _qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    paperless_url = os.environ.get("PAPERLESS_URL")
    paperless_token = _read_secret("PAPERLESS_TOKEN")
    if paperless_url and paperless_token:
        _paperless_client = PaperlessClient(paperless_url, paperless_token)
        log.info("Paperless keyword search enabled (%s)", paperless_url)
    else:
        log.debug("Paperless search disabled (PAPERLESS_URL or PAPERLESS_TOKEN not set)")

    _rerank_model = os.environ.get("RERANK_MODEL") or os.environ.get("METADATA_MODEL") or None
    _rerank_api_base = os.environ.get("RERANK_API_BASE") or os.environ.get("METADATA_API_BASE") or None
    if _rerank_model:
        log.info("LLM reranking enabled (model=%s)", _rerank_model)

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
    if _paperless_client:
        await _paperless_client.aclose()


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


@app.get("/search", response_model=None)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
) -> JSONResponse:
    """Hybrid semantic + keyword search with optional LLM reranking.

    Two-Tower Retrieval:
      - Dense: Local CPU embedding (FastEmbed) → Qdrant cosine search
      - Keyword: Paperless full-text API
      - Merge: Reciprocal Rank Fusion (RRF) to combine incompatible score scales
      - Rerank: LLM-as-a-Judge (optional) filters false positives and reorders

    Returns doc_ids in final rank order (RRF or LLM score, descending).
    Gracefully degrades to dense-only search if Paperless is unavailable.
    """
    if _lazy_embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not ready")

    try:
        # 1. Concurrent dense and keyword search
        keyword_coro = _keyword_search_safe(q) if _paperless_client else _empty_list()

        dense_result, keyword_result = await asyncio.gather(
            dense_search(_lazy_embedder, _qdrant_url, q, K),
            keyword_coro,
            return_exceptions=True,
        )

        if isinstance(dense_result, BaseException):
            log.warning(
                "Search: dense retrieval failed, returning empty results (%s: %s)",
                type(dense_result).__name__,
                dense_result,
            )
            return JSONResponse(content=[])
        dense_results: list[tuple[int, str]] = dense_result
        keyword_ids: list[int] = keyword_result if not isinstance(keyword_result, BaseException) else []

        dense_ids = [doc_id for doc_id, _ in dense_results]
        chunk_map = {doc_id: text for doc_id, text in dense_results}

        # 2. RRF fusion or fallback to dense
        if keyword_ids:
            log.info("Search: dense=%d results, keyword=%d results → RRF fusion", len(dense_ids), len(keyword_ids))
            fused_ids = rrf_fuse(dense_ids, keyword_ids, k=RRF_K)
        else:
            log.debug("Search: keyword track unavailable, using dense results only")
            fused_ids = dense_ids

        # 3. Optional LLM reranking on top N candidates
        if _rerank_model and fused_ids:
            try:
                candidates = [
                    ScoredDoc(doc_id, 0.0, chunk_map.get(doc_id))
                    for doc_id in fused_ids[:N]
                ]
                fused_ids = await llm_rerank(q, candidates, _rerank_model, _rerank_api_base, N)
                log.info("Search: reranked %d candidates, final=%d", len(candidates), len(fused_ids))
            except Exception as e:
                log.warning("LLM reranking failed, using RRF order: %s", e)

        return JSONResponse(content=fused_ids[:limit])

    except (SystemExit, KeyboardInterrupt, GeneratorExit):
        raise
    except BaseException as exc:
        log.warning("Search endpoint error (%s: %s) — returning empty results", type(exc).__name__, exc)
        return JSONResponse(content=[])


async def _empty_list() -> list[int]:
    return []


async def _keyword_search_safe(query: str) -> list[int]:
    """Wrapper around keyword_search() that treats errors as empty results."""
    if _paperless_client is None:
        return []
    try:
        return await _paperless_client.search_documents(query, page_size=N)
    except Exception as e:
        log.warning("Keyword search failed: %s", e)
        return []


@app.get("/health")
async def health() -> dict:
    if _queues:
        pending = await _queues.pending_count()
    else:
        pending = {"ocr": 0, "metadata": 0, "embed": 0}
    return {"status": "ok", "pending": pending}
