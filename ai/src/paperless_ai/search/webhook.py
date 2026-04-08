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
      Headers:
        X-Webhook-Token: <WEBHOOK_SECRET value>

Routing (tag-driven):
  ai:run-ocr      → queue:ocr      (vision OCR stage)
  ai:run-metadata → queue:metadata (LLM metadata extraction stage)
  ai:run-embed    → queue:embed    (embedding stage)
  (no ai:run-*)   → ignored        (no implicit re-index on unrelated updates)

Authentication is optional: if WEBHOOK_SECRET is set, the endpoint validates
the X-Webhook-Token header using constant-time comparison.

Health endpoint:
    GET /health → {"status": "ok", "pending": {"ocr": N, "metadata": N, "embed": N}}
"""

import asyncio
import json
import logging
import os
import re
import secrets
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.core.telemetry import setup_telemetry
from paperless_ai.search.chat_agent import ChatCopilot
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.queue import TaskQueues
from paperless_ai.search.retriever import (
    SearchFilters,
    hybrid_retrieve,
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
_chat_copilot: ChatCopilot | None = None

# Retrieval hyperparameters
K = 25  # max chunks from dense search
N = 50  # min candidate pool size before local reranking
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
    global _chat_copilot

    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _webhook_secret = _read_secret("WEBHOOK_SECRET") or None
    _tag_ocr = os.environ.get("TAG_OCR", os.environ.get("TAG_PENDING", "ai:run-ocr"))
    _tag_metadata = os.environ.get("TAG_METADATA", "ai:run-metadata")
    _tag_embed = os.environ.get("TAG_EMBED", "ai:run-embed")

    # Hybrid search configuration
    _qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    paperless_url = os.environ.get("PAPERLESS_URL")
    paperless_token = _read_secret("PAPERLESS_TOKEN")
    log.info(
        "Startup config: redis=%s qdrant=%s paperless_url=%r paperless_token=%s webhook_secret=%s",
        redis_url,
        _qdrant_url,
        paperless_url,
        "loaded" if paperless_token else "missing",
        "loaded" if _webhook_secret else "missing",
    )
    if paperless_url and paperless_token:
        _paperless_client = PaperlessClient(paperless_url, paperless_token)
        log.info("Paperless keyword search enabled (%s)", paperless_url)
    else:
        log.warning(
            "Paperless search disabled (paperless_url=%s paperless_token=%s)",
            "set" if paperless_url else "missing",
            "set" if paperless_token else "missing",
        )

    log.info(
        "Local reranking enabled (model=%s)",
        LocalLazySearchEmbedder.LOCAL_RERANKER_MODEL_NAME,
    )

    config = AgentConfig.from_env()
    setup_telemetry(service_name=config.name, project_name=config.name)

    if _webhook_secret:
        log.info("Webhook authentication enabled")
    else:
        log.warning("WEBHOOK_SECRET not set — webhook endpoint is unauthenticated")

    _queues = TaskQueues(redis_url)
    _lazy_embedder = LocalLazySearchEmbedder()
    _idle_task = asyncio.create_task(_lazy_embedder.idle_watcher())
    if _paperless_client is not None:
        _chat_copilot = ChatCopilot(
            config,
            _paperless_client,
            _lazy_embedder,
            _qdrant_url,
        )
        log.info("Chat copilot enabled")
    else:
        _chat_copilot = None
        log.warning("Chat copilot disabled (Paperless client unavailable)")
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
    _chat_copilot = None


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


def _route_to_stage(tags: set[str]) -> str | None:
    """Determine which queue stage to route to based on document tags.

    Priority: ocr > metadata > embed. Untagged updates are ignored.
    """
    if _tag_ocr in tags:
        return TaskQueues.KEY_OCR
    if _tag_metadata in tags:
        return TaskQueues.KEY_METADATA
    if _tag_embed in tags:
        return TaskQueues.KEY_EMBED
    return None


def _parse_tags(body: dict) -> set[str]:
    """Parse tag names from an optional webhook payload field.

    Some tests and custom callers provide a comma-separated list of tag names.
    Paperless workflow webhooks on this deployment do not expose tags, so the
    production routing path resolves current tags from the Paperless API.
    """
    raw = body.get("tag_list", body.get("document_tags", ""))
    if not raw:
        return set()
    return {t.strip() for t in str(raw).split(",") if t.strip()}


async def _get_current_document_tags(doc_id: int, payload_tags: set[str]) -> set[str]:
    """Return current tag names from Paperless, falling back to payload tags."""
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
    except Exception as e:
        log.warning("Webhook: failed to resolve current tags for document %d: %s", doc_id, e)
        return payload_tags


async def _refresh_qdrant_payload(doc_id: int) -> bool:
    """Refresh Qdrant payload metadata for an already-indexed document."""
    if _paperless_client is None:
        return False

    try:
        doc = await _paperless_client.get_document(doc_id)
        if doc is None:
            return False

        from paperless_ai.core.runner import _build_search_metadata
        from paperless_ai.search.qdrant_store import QdrantDocumentStore

        tag_embed_name = os.environ.get("TAG_EMBED", "ai:run-embed")
        tag_embed_id = None
        try:
            tag_embed_id = await _paperless_client.get_tag_id(tag_embed_name, create=False)
        except ValueError:
            tag_embed_id = None

        meta = await _build_search_metadata(
            _paperless_client,
            doc,
            title=doc.get("title"),
            correspondent=await _paperless_client.get_correspondent_name(doc["correspondent"])
            if doc.get("correspondent")
            else None,
            document_date=doc.get("created"),
            summary=None,
            exclude_tag_ids={tag_embed_id} if tag_embed_id is not None else None,
        )

        store = QdrantDocumentStore(_qdrant_url)
        try:
            await store.update_document_payload(
                doc_id=doc_id,
                title=meta.title,
                correspondent=meta.correspondent,
                document_type=meta.document_type,
                storage_path=meta.storage_path,
                tags=meta.tags,
                date=meta.document_date,
                year=meta.year,
            )
        finally:
            await store.aclose()
        return True
    except Exception as e:
        log.warning("Webhook: failed to refresh Qdrant payload for document %d: %s", doc_id, e)
        return False


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
        return Response(status_code=202)  # Accept anyway — don't make Paperless retry

    if _queues:
        tags = await _get_current_document_tags(doc_id, _parse_tags(body))
        stage = _route_to_stage(tags)
        log.info(
            "Webhook routing: document %d tags=%s stage=%s",
            doc_id,
            sorted(tags),
            stage.split(":")[-1] if stage else "none",
        )
        if stage is None:
            refreshed = await _refresh_qdrant_payload(doc_id)
            if refreshed:
                log.info("Webhook result: document %d refreshed Qdrant payload only", doc_id)
            else:
                log.info("Webhook result: document %d ignored (no ai:run-* tags present)", doc_id)
        else:
            added = await _queues.enqueue(doc_id, stage)
            log.info(
                "Webhook result: document %d → %s (%s)",
                doc_id,
                stage.split(":")[-1],
                "queued" if added else "already pending",
            )

    return Response(status_code=202)


@app.get("/search", response_model=None)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    correspondent: str | None = Query(None),
    document_type: str | None = Query(None),
    storage_path: str | None = Query(None),
    tags: list[str] | None = Query(None),
    year: str | None = Query(None),
) -> JSONResponse:
    """Hybrid semantic + keyword search with local BGE reranking.

    Two-Tower Retrieval:
      - Dense: Local CPU embedding (FastEmbed) → Qdrant cosine search
      - Keyword: Paperless full-text API
      - Merge: Reciprocal Rank Fusion (RRF) to combine incompatible score scales
      - Rerank: local bge-reranker-v2-m3 reorders fused candidates

    Returns doc_ids in final rank order (reranker score, descending).
    Gracefully degrades to dense-only search if Paperless is unavailable.
    """
    if _lazy_embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not ready")

    try:
        filters = SearchFilters(
            correspondent=correspondent,
            document_type=document_type,
            storage_path=storage_path,
            tags=tags,
            year=year,
        )
        try:
            fused_ids, _chunk_map = await hybrid_retrieve(
                embedder=_lazy_embedder,
                qdrant_url=_qdrant_url,
                query=q,
                client=_paperless_client,
                filters=filters,
                dense_k=K,
                rerank_candidates=max(N, limit),
                rrf_k=RRF_K,
            )
        except Exception as exc:
            log.warning(
                "Search: hybrid retrieval failed, returning empty results (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return JSONResponse(content=[])

        return JSONResponse(content=fused_ids[:limit])

    except (SystemExit, KeyboardInterrupt, GeneratorExit):
        raise
    except BaseException as exc:
        log.warning("Search endpoint error (%s: %s) — returning empty results", type(exc).__name__, exc)
        return JSONResponse(content=[])


@app.get("/metadata/available")
async def available_metadata() -> JSONResponse:
    """Return exact metadata names available for agentic search pre-filtering."""
    if _paperless_client is None:
        raise HTTPException(status_code=503, detail="Paperless client not configured")
    return JSONResponse(content=await _paperless_client.get_available_metadata())


def _document_detail_url(doc_id: int) -> str:
    return f"/documents/{doc_id}/detail"


def _document_thumb_url(doc_id: int) -> str:
    return f"/api/documents/{doc_id}/thumb/"


def _document_preview_url(doc_id: int) -> str:
    return f"/api/documents/{doc_id}/preview/"


async def _build_chat_sources(source_flags: dict[int, dict[str, bool]]) -> list[dict]:
    if _paperless_client is None:
        return []

    items: list[dict] = []
    for doc_id, flags in sorted(
        source_flags.items(),
        key=lambda item: (not item[1].get("inspected", False), item[0]),
    ):
        metadata = await _paperless_client.get_document_chat_metadata(doc_id)
        if metadata is None:
            continue
        items.append(
            {
                **metadata,
                "matched": bool(flags.get("matched")),
                "inspected": bool(flags.get("inspected")),
                "detail_url": _document_detail_url(doc_id),
                "thumb_url": _document_thumb_url(doc_id),
                "preview_url": _document_preview_url(doc_id),
            }
        )
    return items


@app.get("/chat")
async def chat_ui() -> HTMLResponse:
    """Serve the browser UI for the Paperless copilot."""
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paperless Copilot</title>
  <link href="/static/bootstrap.min.css" rel="stylesheet">
  <link href="/static/base.css" rel="stylesheet">
  <style>
    :root {
      color-scheme: light;
      --chat-bg: #f5f5f5;
      --chat-panel: #ffffff;
      --chat-muted: #6c757d;
      --chat-border: #dee2e6;
      --chat-primary: #17541f;
      --chat-primary-soft: #d9eadc;
      --chat-shadow: 0 0.75rem 1.5rem rgba(33, 37, 41, 0.08);
      --chat-radius: 0.85rem;
    }
    body {
      margin: 0;
      background: var(--chat-bg);
      color: var(--bs-body-color);
      font-family: var(--bs-body-font-family);
    }
    .chat-shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 1.25rem;
    }
    .chat-header {
      margin-bottom: 1rem;
      border: 1px solid var(--chat-border);
      border-radius: var(--chat-radius);
      box-shadow: var(--chat-shadow);
      background: linear-gradient(180deg, #ffffff, #fbfcfb);
    }
    .chat-layout {
      display: grid;
      gap: 1rem;
      grid-template-columns: minmax(0, 2.25fr) minmax(300px, 1fr);
      align-items: start;
    }
    .chat-column,
    .preview-column {
      min-width: 0;
    }
    .chat-panel,
    .preview-panel {
      border: 1px solid var(--chat-border);
      border-radius: var(--chat-radius);
      box-shadow: var(--chat-shadow);
      background: var(--chat-panel);
    }
    .socket-banner {
      display: none;
      margin-bottom: 0.75rem;
    }
    .socket-banner.active {
      display: block;
    }
    .conversation {
      display: grid;
      gap: 0.9rem;
      padding: 1rem;
      min-height: 58vh;
    }
    .bubble {
      max-width: min(78ch, 100%);
      padding: 0.9rem 1rem;
      border-radius: 1rem;
      white-space: pre-wrap;
      box-shadow: 0 0.35rem 0.8rem rgba(0, 0, 0, 0.05);
    }
    .bubble.user {
      justify-self: end;
      background: var(--chat-primary);
      color: #fff;
      border-bottom-right-radius: 0.3rem;
    }
    .turn {
      display: grid;
      gap: 0.75rem;
    }
    .bubble.assistant {
      background: #fff;
      border: 1px solid var(--chat-border);
      border-bottom-left-radius: 0.3rem;
    }
    .bubble.assistant.is-pending {
      color: var(--chat-muted);
    }
    .turn-meta {
      display: grid;
      gap: 0.75rem;
      margin-left: 0.25rem;
    }
    .timeline {
      display: grid;
      gap: 0.45rem;
      padding-left: 0.25rem;
    }
    .timeline-item {
      color: var(--chat-muted);
      font-size: 0.93rem;
    }
    .usage {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .usage-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      background: var(--bs-light-bg-subtle);
      color: var(--bs-secondary-text-emphasis);
      font-size: 0.82rem;
      border: 1px solid var(--chat-border);
    }
    .tools-panel,
    .sources-panel {
      border: 1px solid var(--chat-border);
      border-radius: 0.8rem;
      background: #fcfcfc;
    }
    .tools-panel summary,
    .sources-panel summary {
      cursor: pointer;
      list-style: none;
      padding: 0.8rem 0.95rem;
      font-weight: 600;
      color: var(--chat-primary);
    }
    .tools-panel summary::-webkit-details-marker,
    .sources-panel summary::-webkit-details-marker {
      display: none;
    }
    .tool-list {
      display: grid;
      gap: 0.75rem;
      padding: 0 0.95rem 0.95rem;
    }
    .tool-card {
      border: 1px solid var(--chat-border);
      border-radius: 0.8rem;
      background: #fff;
      overflow: hidden;
    }
    .tool-card details summary {
      padding: 0.7rem 0.85rem;
      background: #f8f9fa;
      color: var(--bs-body-color);
      font-weight: 600;
    }
    .tool-body {
      padding: 0.8rem 0.85rem 0.9rem;
      display: grid;
      gap: 0.65rem;
    }
    .tool-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-size: 0.82rem;
      color: var(--chat-muted);
    }
    .tool-preview,
    .tool-arguments {
      margin: 0;
      font-size: 0.9rem;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .source-list {
      display: grid;
      gap: 0.85rem;
      padding: 0 0.95rem 0.95rem;
    }
    .source-card {
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr);
      gap: 0.9rem;
      border: 1px solid var(--chat-border);
      border-radius: 0.85rem;
      background: #fff;
      overflow: hidden;
    }
    .source-thumb {
      width: 108px;
      height: 132px;
      object-fit: cover;
      background: #edf1ed;
      border-right: 1px solid var(--chat-border);
    }
    .source-content {
      padding: 0.8rem 0.9rem 0.85rem 0;
      display: grid;
      gap: 0.55rem;
      min-width: 0;
    }
    .source-title {
      margin: 0;
      font-size: 1rem;
      font-weight: 600;
    }
    .source-title a {
      color: inherit;
      text-decoration: none;
    }
    .source-title a:hover {
      color: var(--chat-primary);
    }
    .source-badges,
    .source-meta,
    .source-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .source-badge,
    .source-meta span {
      display: inline-flex;
      align-items: center;
      padding: 0.22rem 0.5rem;
      border-radius: 999px;
      font-size: 0.78rem;
      border: 1px solid var(--chat-border);
      background: #f8f9fa;
      color: var(--bs-secondary-text-emphasis);
    }
    .source-badge.match {
      background: #eaf3eb;
      color: var(--chat-primary);
      border-color: #cfe0d1;
    }
    .source-badge.read {
      background: #fff3cd;
      color: #664d03;
      border-color: #ffe69c;
    }
    .composer {
      display: grid;
      gap: 0.75rem;
      padding: 1rem;
      border-top: 1px solid var(--chat-border);
      background: linear-gradient(180deg, rgba(248,249,250,0.25), rgba(255,255,255,0.95));
    }
    .composer textarea {
      width: 100%;
      min-height: 5.5rem;
      resize: vertical;
      border: 1px solid var(--chat-border);
      border-radius: 0.85rem;
      padding: 0.85rem 0.95rem;
      font: inherit;
      background: #fff;
    }
    .composer-actions {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }
    .composer-hint {
      color: var(--chat-muted);
      font-size: 0.9rem;
    }
    .preview-panel {
      position: sticky;
      top: 1.25rem;
      min-height: 18rem;
      overflow: hidden;
    }
    .preview-placeholder {
      padding: 1.1rem;
      color: var(--chat-muted);
    }
    .preview-header {
      display: none;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      padding: 1rem 1rem 0.75rem;
      border-bottom: 1px solid var(--chat-border);
    }
    .preview-header.active {
      display: flex;
    }
    .preview-frame-wrap {
      display: none;
      padding: 1rem;
      background: #eef1ef;
    }
    .preview-frame-wrap.active {
      display: block;
    }
    .preview-frame {
      width: 100%;
      min-height: 72vh;
      border: 0;
      border-radius: 0.7rem;
      background: #fff;
    }
    @media (max-width: 1080px) {
      .chat-layout {
        grid-template-columns: 1fr;
      }
      .preview-panel {
        position: static;
      }
      .preview-frame {
        min-height: 55vh;
      }
    }
    @media (max-width: 720px) {
      .chat-shell {
        padding: 0.75rem;
      }
      .conversation,
      .composer {
        padding: 0.85rem;
      }
      .source-card {
        grid-template-columns: 1fr;
      }
      .source-thumb {
        width: 100%;
        height: 160px;
        border-right: 0;
        border-bottom: 1px solid var(--chat-border);
      }
      .source-content {
        padding: 0 0.85rem 0.85rem;
      }
      .composer-actions {
        align-items: stretch;
        flex-direction: column;
      }
    }
  </style>
</head>
<body>
  <main class="chat-shell">
    <section class="chat-header card">
      <div class="card-body p-4">
        <div class="d-flex flex-wrap justify-content-between align-items-start gap-3">
          <div>
            <p class="text-uppercase text-muted fw-semibold small mb-2">Paperless-ngx Copilot</p>
            <h1 class="h3 mb-2">Search your archive conversationally</h1>
            <p class="text-muted mb-0">Ask about invoices, receipts, tags, correspondents, or specific facts. The assistant streams progress, shows tool calls, and cites source documents.</p>
          </div>
          <div class="usage">
            <span class="usage-chip">Transport: WebSocket</span>
            <span class="usage-chip">UI: Vanilla JS</span>
          </div>
        </div>
      </div>
    </section>

    <div class="chat-layout">
      <section class="chat-column">
        <div id="socket-banner" class="socket-banner alert alert-warning mb-0"></div>
        <section class="chat-panel">
          <div id="conversation" class="conversation"></div>
          <form id="chat-form" class="composer">
            <textarea id="prompt" placeholder="Ask about invoices from 2024, documents from a correspondent, or the contents of a specific receipt..."></textarea>
            <div class="composer-actions">
              <div class="composer-hint">Tool details stay collapsed by default so the transcript remains easy to demo.</div>
              <button id="send-button" type="submit" class="btn btn-primary px-4">Send</button>
            </div>
          </form>
        </section>
      </section>

      <aside class="preview-column">
        <section id="preview-panel" class="preview-panel">
          <div id="preview-placeholder" class="preview-placeholder">
            Select a source card to open an inline preview here. Use the Paperless link on the card for the full document view.
          </div>
          <div id="preview-header" class="preview-header">
            <div>
              <h2 id="preview-title" class="h6 mb-1">Document preview</h2>
              <p id="preview-subtitle" class="text-muted small mb-0"></p>
            </div>
            <a id="preview-open-link" class="btn btn-sm btn-outline-secondary" target="_blank" rel="noreferrer">Open in Paperless</a>
          </div>
          <div id="preview-frame-wrap" class="preview-frame-wrap">
            <iframe id="preview-frame" class="preview-frame" title="Document preview"></iframe>
          </div>
        </section>
      </aside>
    </div>
  </main>
  <script>
    const conversation = document.getElementById("conversation");
    const form = document.getElementById("chat-form");
    const prompt = document.getElementById("prompt");
    const sendButton = document.getElementById("send-button");
    const socketBanner = document.getElementById("socket-banner");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const previewHeader = document.getElementById("preview-header");
    const previewTitle = document.getElementById("preview-title");
    const previewSubtitle = document.getElementById("preview-subtitle");
    const previewOpenLink = document.getElementById("preview-open-link");
    const previewFrameWrap = document.getElementById("preview-frame-wrap");
    const previewFrame = document.getElementById("preview-frame");
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const basePath = window.location.pathname.replace(/\/chat\/?$/, "");
    const wsPath = `${basePath}/ws/chat`.replace(/\/{2,}/g, "/");
    const socket = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);
    const turns = new Map();

    function scrollConversation() {
      conversation.scrollTop = conversation.scrollHeight;
    }

    function setSocketBanner(kind, text) {
      socketBanner.className = `socket-banner alert alert-${kind} mb-0 active`;
      socketBanner.textContent = text;
    }

    function clearSocketBanner() {
      socketBanner.className = "socket-banner alert alert-warning mb-0";
      socketBanner.textContent = "";
    }

    function addUserBubble(content) {
      const bubble = document.createElement("div");
      bubble.className = "bubble user";
      bubble.textContent = content;
      conversation.appendChild(bubble);
      scrollConversation();
    }

    function createTurn(turnId) {
      const turn = document.createElement("article");
      turn.className = "turn";
      turn.dataset.turnId = turnId;

      const answer = document.createElement("div");
      answer.className = "bubble assistant is-pending";
      answer.textContent = "Waiting for the assistant…";

      const meta = document.createElement("div");
      meta.className = "turn-meta";

      const timeline = document.createElement("div");
      timeline.className = "timeline";

      const usage = document.createElement("div");
      usage.className = "usage";

      const toolsPanel = document.createElement("details");
      toolsPanel.className = "tools-panel";
      const toolsSummary = document.createElement("summary");
      toolsSummary.textContent = "Tool calls";
      const toolList = document.createElement("div");
      toolList.className = "tool-list";
      toolsPanel.appendChild(toolsSummary);
      toolsPanel.appendChild(toolList);

      const sourcesPanel = document.createElement("details");
      sourcesPanel.className = "sources-panel";
      const sourcesSummary = document.createElement("summary");
      sourcesSummary.textContent = "Sources";
      const sourceList = document.createElement("div");
      sourceList.className = "source-list";
      sourcesPanel.appendChild(sourcesSummary);
      sourcesPanel.appendChild(sourceList);

      toolsPanel.hidden = true;
      sourcesPanel.hidden = true;

      meta.appendChild(timeline);
      meta.appendChild(usage);
      meta.appendChild(toolsPanel);
      meta.appendChild(sourcesPanel);
      turn.appendChild(answer);
      turn.appendChild(meta);
      conversation.appendChild(turn);

      const state = {
        root: turn,
        answer,
        timeline,
        usage,
        toolsPanel,
        toolList,
        toolsSummary,
        sourcesPanel,
        sourceList,
        sourcesSummary,
        toolEntries: new Map(),
      };
      turns.set(turnId, state);
      scrollConversation();
      return state;
    }

    function getTurn(turnId) {
      return turns.get(turnId) || createTurn(turnId);
    }

    function addTimelineItem(turnId, text) {
      const turn = getTurn(turnId);
      const item = document.createElement("div");
      item.className = "timeline-item";
      item.textContent = text;
      turn.timeline.appendChild(item);
      scrollConversation();
    }

    function renderUsageChip(label, value) {
      const chip = document.createElement("span");
      chip.className = "usage-chip";
      chip.textContent = `${label}: ${value}`;
      return chip;
    }

    function updateUsage(turnId, payload) {
      const turn = getTurn(turnId);
      turn.usage.innerHTML = "";
      if (!payload.available) {
        turn.usage.appendChild(renderUsageChip("Tokens", "n/a"));
        return;
      }
      turn.usage.appendChild(renderUsageChip("Prompt", payload.prompt_tokens || 0));
      turn.usage.appendChild(renderUsageChip("Completion", payload.completion_tokens || 0));
      turn.usage.appendChild(renderUsageChip("Total", payload.total_tokens || 0));
      if (payload.model) {
        turn.usage.appendChild(renderUsageChip("Model", payload.model));
      }
    }

    function formatJson(value) {
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    }

    function getOrCreateToolCard(turnId, payload) {
      const turn = getTurn(turnId);
      turn.toolsPanel.hidden = false;
      const existing = turn.toolEntries.get(payload.tool_call_id);
      if (existing) {
        return existing;
      }

      const card = document.createElement("div");
      card.className = "tool-card";
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = payload.name;
      const body = document.createElement("div");
      body.className = "tool-body";
      const meta = document.createElement("div");
      meta.className = "tool-meta";
      const preview = document.createElement("pre");
      preview.className = "tool-preview";
      const args = document.createElement("pre");
      args.className = "tool-arguments";
      args.textContent = formatJson(payload.arguments || {});
      body.appendChild(meta);
      body.appendChild(preview);
      body.appendChild(args);
      details.appendChild(summary);
      details.appendChild(body);
      card.appendChild(details);
      turn.toolList.appendChild(card);
      turn.toolEntries.set(payload.tool_call_id, { card, details, summary, meta, preview, args });
      turn.toolsSummary.textContent = `Tool calls (${turn.toolEntries.size})`;
      return turn.toolEntries.get(payload.tool_call_id);
    }

    function updateTool(turnId, payload, started) {
      const tool = getOrCreateToolCard(turnId, payload);
      tool.summary.textContent = payload.name;
      if (started) {
        tool.meta.innerHTML = "<span>Running…</span>";
        tool.preview.textContent = "";
      } else {
        tool.meta.innerHTML = "";
        const pieces = [];
        if (payload.duration_ms != null) {
          pieces.push(`Duration ${payload.duration_ms} ms`);
        }
        if (payload.summary) {
          pieces.push(payload.summary);
        }
        pieces.forEach((text) => {
          const span = document.createElement("span");
          span.textContent = text;
          tool.meta.appendChild(span);
        });
        tool.preview.textContent = payload.preview || "";
      }
    }

    function renderSourceBadges(source) {
      const badges = [];
      if (source.matched) badges.push(["Matched", "match"]);
      if (source.inspected) badges.push(["Read in full", "read"]);
      if (source.document_type_name) badges.push([source.document_type_name, ""]);
      return badges;
    }

    function openPreview(source) {
      previewPlaceholder.hidden = true;
      previewHeader.classList.add("active");
      previewFrameWrap.classList.add("active");
      previewTitle.textContent = source.title || `Document ${source.id}`;
      previewSubtitle.textContent = `Document ${source.id}${source.original_filename ? ` · ${source.original_filename}` : ""}`;
      previewOpenLink.href = source.detail_url;
      previewFrame.src = source.preview_url;
    }

    function renderSources(turnId, items) {
      const turn = getTurn(turnId);
      turn.sourcesPanel.hidden = items.length === 0;
      turn.sourcesSummary.textContent = `Sources (${items.length})`;
      turn.sourceList.innerHTML = "";
      items.forEach((source) => {
        const card = document.createElement("article");
        card.className = "source-card";

        const thumb = document.createElement("img");
        thumb.className = "source-thumb";
        thumb.loading = "lazy";
        thumb.alt = source.title || `Document ${source.id}`;
        thumb.src = source.thumb_url;
        thumb.onerror = () => {
          thumb.style.visibility = "hidden";
        };

        const content = document.createElement("div");
        content.className = "source-content";

        const title = document.createElement("h3");
        title.className = "source-title";
        const titleLink = document.createElement("a");
        titleLink.href = source.detail_url;
        titleLink.target = "_blank";
        titleLink.rel = "noreferrer";
        titleLink.textContent = source.title || `Document ${source.id}`;
        title.appendChild(titleLink);

        const badges = document.createElement("div");
        badges.className = "source-badges";
        renderSourceBadges(source).forEach(([label, klass]) => {
          const badge = document.createElement("span");
          badge.className = `source-badge ${klass}`.trim();
          badge.textContent = label;
          badges.appendChild(badge);
        });

        const meta = document.createElement("div");
        meta.className = "source-meta";
        [
          `ID ${source.id}`,
          source.created,
          source.correspondent_name,
          source.storage_path_name,
          source.archive_serial_number ? `ASN ${source.archive_serial_number}` : "",
        ].filter(Boolean).forEach((item) => {
          const span = document.createElement("span");
          span.textContent = item;
          meta.appendChild(span);
        });

        const actions = document.createElement("div");
        actions.className = "source-actions";
        const previewButton = document.createElement("button");
        previewButton.type = "button";
        previewButton.className = "btn btn-sm btn-outline-secondary";
        previewButton.textContent = "Preview";
        previewButton.addEventListener("click", () => openPreview(source));

        const openLink = document.createElement("a");
        openLink.className = "btn btn-sm btn-link px-0";
        openLink.href = source.detail_url;
        openLink.target = "_blank";
        openLink.rel = "noreferrer";
        openLink.textContent = "Open in Paperless";

        actions.appendChild(previewButton);
        actions.appendChild(openLink);

        content.appendChild(title);
        content.appendChild(badges);
        content.appendChild(meta);
        content.appendChild(actions);
        card.appendChild(thumb);
        card.appendChild(content);
        turn.sourceList.appendChild(card);
      });
    }

    function setAssistantMessage(turnId, content) {
      const turn = getTurn(turnId);
      turn.answer.classList.remove("is-pending");
      turn.answer.textContent = content || "(no response)";
      scrollConversation();
    }

    function setTurnError(turnId, content) {
      const turn = getTurn(turnId);
      turn.answer.classList.remove("is-pending");
      turn.answer.textContent = content;
      turn.answer.classList.add("border", "border-danger-subtle", "bg-danger-subtle");
      scrollConversation();
    }

    socket.addEventListener("open", () => {
      clearSocketBanner();
      sendButton.disabled = false;
    });

    socket.addEventListener("close", () => {
      setSocketBanner("warning", "Connection closed. Reload the page to reconnect.");
      sendButton.disabled = true;
    });

    socket.addEventListener("error", () => {
      setSocketBanner("danger", "The chat connection encountered an error.");
      sendButton.disabled = true;
    });

    socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      const turnId = payload.turn_id;
      if (payload.type === "turn_started") {
        createTurn(turnId);
        addTimelineItem(turnId, "Turn started.");
        return;
      }
      if (payload.type === "status") {
        addTimelineItem(turnId, payload.content || "Working…");
        return;
      }
      if (payload.type === "tool_call_started") {
        updateTool(turnId, payload, true);
        return;
      }
      if (payload.type === "tool_call_completed") {
        updateTool(turnId, payload, false);
        return;
      }
      if (payload.type === "usage" && payload.scope === "total") {
        updateUsage(turnId, payload);
        return;
      }
      if (payload.type === "assistant_message") {
        setAssistantMessage(turnId, payload.content);
        return;
      }
      if (payload.type === "sources") {
        renderSources(turnId, payload.items || []);
        return;
      }
      if (payload.type === "error") {
        setTurnError(turnId, payload.content || "Chat failed.");
        return;
      }
      if (payload.type === "turn_completed") {
        addTimelineItem(turnId, payload.success ? "Answer ready." : "Turn failed.");
      }
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const content = prompt.value.trim();
      if (!content) return;
      if (socket.readyState !== WebSocket.OPEN) {
        setSocketBanner("warning", "The connection is not open. Reload the page and try again.");
        return;
      }
      addUserBubble(content);
      socket.send(content);
      prompt.value = "";
      prompt.focus();
    });

    prompt.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey) {
        return;
      }
      event.preventDefault();
      form.requestSubmit();
    });
  </script>
</body>
</html>
        """
    )


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Interactive Paperless copilot over WebSocket."""
    await websocket.accept()
    if _chat_copilot is None:
        await websocket.send_json(
            {
                "type": "error",
                "turn_id": "unavailable",
                "content": "Chat is unavailable because Paperless is not configured.",
            }
        )
        await websocket.close(code=1011)
        return

    history: list[dict] = []
    try:
        while True:
            user_message = (await websocket.receive_text()).strip()
            if not user_message:
                continue
            turn_id = uuid.uuid4().hex
            await websocket.send_json({"type": "turn_started", "turn_id": turn_id})

            async def emit(event: dict) -> None:
                payload = {"turn_id": turn_id, **event}
                await websocket.send_json(payload)

            try:
                result = await _chat_copilot.run_turn(user_message, history, event_callback=emit)
            except Exception as exc:
                log.exception("Chat turn failed")
                await emit({"type": "error", "content": f"Chat request failed: {type(exc).__name__}: {exc}"})
                await emit({"type": "turn_completed", "success": False})
                continue
            history = result.history
            await emit({"type": "assistant_message", "content": result.reply or "(no response)"})
            await emit(
                {
                    "type": "usage",
                    "scope": "total",
                    "model": _chat_copilot._config.chat_model,
                    "available": bool(result.usage),
                    **(result.usage or {}),
                }
            )
            await emit({"type": "sources", "items": await _build_chat_sources(result.sources)})
            await emit({"type": "turn_completed", "success": True})
    except WebSocketDisconnect:
        return


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
