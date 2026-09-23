"""Interactive copilot and search app hosted by the always-on AI service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import niquests
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse, JSONResponse
from paperless_common.secrets import read_secret

from paperless_ai.core.config import AgentConfig
from paperless_common.paperless import PaperlessClient, _raise_for_status
from paperless_common.telemetry import setup_telemetry
from paperless_common.queue import TaskQueues
from paperless_ai.search.retriever import (
    SearchFilters,
    hybrid_retrieve,
)

if TYPE_CHECKING:
    from paperless_ai.search.chat_agent import ChatCopilot
    from paperless_ai.search.chat_store import ChatStore
    from paperless_ai.search.local_search_process import ProcessLocalSearchEmbedder

log = logging.getLogger(__name__)

_queues: TaskQueues | None = None
_lazy_embedder: ProcessLocalSearchEmbedder | None = None
_paperless_client: PaperlessClient | None = None
_qdrant_store = None
_qdrant_url: str = "http://qdrant:6333"
_chat_copilot: ChatCopilot | None = None
_chat_store: ChatStore | None = None
_config: AgentConfig | None = None
_local_search_idle_timeout_seconds: int = 300
_local_search_start_method: str = "spawn"
_local_search_warm_on_startup: bool = False
_local_search_warm_on_chat_load: bool = True
_local_search_warmup_task: asyncio.Task | None = None
_worker_tasks: list[asyncio.Task] = []
_worker_heartbeats: dict[str, float] = {
    "ocr": 0.0,
    "metadata": 0.0,
    "embed": 0.0,
    "refresh": 0.0,
}
_worker_ready: bool = False
_worker_setup_error: str | None = None
_search_request_timeout_seconds: float = 20.0

# Retrieval hyperparameters
K = 25  # max chunks from dense search
N = 50  # min candidate pool size before local reranking
RRF_K = 60  # RRF smoothing constant


def _is_retryable_paperless_error(exc: Exception) -> bool:
    """Return whether a Paperless startup error may be transient."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {408, 429} or status_code >= 500
    return isinstance(exc, niquests.RequestException)


def _is_unavailable_chat_source_error(exc: Exception) -> bool:
    """Return whether a document metadata failure should retain an unavailable source."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {401, 403, 404, 408, 429} or status_code >= 500
    return isinstance(exc, niquests.RequestException)


async def _initialize_paperless(
    client: PaperlessClient,
    config: AgentConfig,
    *,
    retry_delay: float = 1.0,
    max_retry_delay: float = 60.0,
) -> tuple[int, int, int]:
    """Wait for Paperless and ensure the AI-managed resources are ready."""
    attempt = 0
    while True:
        attempt += 1
        try:
            response = await client._client.get("/api/")
            _raise_for_status(response)
            log.info(
                "Paperless API reachable (version: %s)",
                response.headers.get("x-version", "unknown"),
            )

            if config.manage_paperless_workflows:
                added_wf_id, updated_wf_id = await client.ensure_ai_workflows(
                    tag_ocr=config.tag_ocr,
                    webhook_url=config.paperless_webhook_url,
                    webhook_secret=config.webhook_secret,
                )
                log.info(
                    "Paperless workflows ready: document_added=%d document_updated=%d",
                    added_wf_id,
                    updated_wf_id,
                )
            else:
                added_wf_id = updated_wf_id = 0

            custom_field_id = await client.get_or_create_custom_field(
                "ai_processed", data_type="date"
            )
            ai_summary_field_id = await client.get_or_create_custom_field(
                "ai_summary", data_type="longtext"
            )
            ai_result_field_id = await client.get_or_create_custom_field(
                "ai_result", data_type="longtext"
            )
            log.info(
                "Custom fields: ai_processed=%d ai_summary=%d ai_result=%d",
                custom_field_id,
                ai_summary_field_id,
                ai_result_field_id,
            )
            return custom_field_id, ai_summary_field_id, ai_result_field_id
        except Exception as exc:
            if not _is_retryable_paperless_error(exc):
                raise
            delay = min(max_retry_delay, retry_delay * 2 ** min(attempt - 1, 6))
            log.warning(
                "Paperless startup attempt %d failed: %s; retrying in %.1fs",
                attempt,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global \
        _queues, \
        _lazy_embedder, \
        _paperless_client, \
        _qdrant_store, \
        _qdrant_url, \
        _chat_copilot, \
        _chat_store, \
        _config
    global _local_search_idle_timeout_seconds, _local_search_start_method
    global _local_search_warm_on_startup, _local_search_warm_on_chat_load
    global _local_search_warmup_task
    global _worker_tasks, _worker_heartbeats, _worker_ready, _worker_setup_error
    global _search_request_timeout_seconds

    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    _qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    paperless_url = os.environ.get("PAPERLESS_URL")
    paperless_token = read_secret("PAPERLESS_TOKEN")
    _local_search_idle_timeout_seconds = int(
        os.environ.get("LOCAL_SEARCH_IDLE_TIMEOUT_SECONDS", "300")
    )
    _local_search_start_method = os.environ.get("LOCAL_SEARCH_START_METHOD", "spawn")
    _local_search_warm_on_startup = (
        os.environ.get("LOCAL_SEARCH_WARM_ON_STARTUP", "false").lower() == "true"
    )
    _local_search_warm_on_chat_load = (
        os.environ.get("LOCAL_SEARCH_WARM_ON_CHAT_LOAD", "true").lower() == "true"
    )
    _search_request_timeout_seconds = float(
        os.environ.get("SEARCH_REQUEST_TIMEOUT_SECONDS", "20")
    )
    log.info(
        "Startup config: redis=%s qdrant=%s paperless_url=%r paperless_token=%s local_search_idle_timeout=%ss local_search_start_method=%s warm_on_startup=%s warm_on_chat_load=%s",
        redis_url,
        _qdrant_url,
        paperless_url,
        "loaded" if paperless_token else "missing",
        _local_search_idle_timeout_seconds,
        _local_search_start_method,
        _local_search_warm_on_startup,
        _local_search_warm_on_chat_load,
    )
    if not paperless_url:
        raise RuntimeError("PAPERLESS_URL is not set for the copilot service")
    if not paperless_token:
        raise RuntimeError(
            "PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set for the copilot service"
        )

    _paperless_client = PaperlessClient(paperless_url, paperless_token)
    log.info("Paperless keyword search enabled (%s)", paperless_url)

    log.info("Local reranking enabled lazily for search/chat requests")

    _config = AgentConfig.from_env()
    setup_telemetry(service_name=_config.name, project_name=_config.name)
    chat_database_url = os.environ.get("CHAT_DATABASE_URL")
    if not chat_database_url:
        raise RuntimeError("CHAT_DATABASE_URL is not set for the copilot service")
    from paperless_ai.search.chat_store import ChatStore

    _chat_store = ChatStore(
        chat_database_url, password=read_secret("CHAT_DATABASE_PASSWORD")
    )
    await _chat_store.migrate()
    log.info("Chat history database migrations are ready")
    _queues = TaskQueues(redis_url)
    _lazy_embedder = None

    from paperless_ai.core.runner import (
        clear_shutdown_request,
        close_model_probe_session,
        request_shutdown,
        run_embed_batch,
        run_metadata_batch,
        run_ocr_batch,
        run_refresh_batch,
    )
    from paperless_ai.search.embedder import EmbeddingAPIEmbedder
    from paperless_ai.search.qdrant_store import QdrantDocumentStore

    _worker_ready = False
    _worker_setup_error = None
    _worker_heartbeats = {"ocr": 0.0, "metadata": 0.0, "embed": 0.0, "refresh": 0.0}
    _worker_tasks = []
    clear_shutdown_request()

    log.info("Checking Paperless API connectivity and managed resources...")
    (
        custom_field_id,
        ai_summary_field_id,
        ai_result_field_id,
    ) = await _initialize_paperless(_paperless_client, _config)

    store = QdrantDocumentStore(_config.qdrant_url)
    try:
        await store.ensure_collection()
        log.info("Qdrant collection ready (%s)", _config.qdrant_url)
    except Exception as exc:
        log.warning("Qdrant not reachable: %s — embedding will be skipped", exc)
        store = None
    _qdrant_store = store
    _chat_copilot = None
    log.info("Chat copilot enabled lazily")

    embedder = EmbeddingAPIEmbedder(_config.embedding_endpoint, _config.embedding_model)
    if not await embedder.check_connectivity():
        log.warning(
            "Embedding API not reachable at %s — embedding will be skipped",
            _config.embedding_endpoint,
        )
        await embedder.aclose()
        embedder = None

    def _mark_worker_heartbeat(stage: str) -> None:
        _worker_heartbeats[stage] = time.time()

    async def _sleep_or_stop() -> bool:
        try:
            await asyncio.sleep(_config.poll_interval)
            return False
        except asyncio.CancelledError:
            return True

    async def _ocr_worker() -> None:
        while True:
            try:
                success, failure = await run_ocr_batch(
                    _paperless_client, _config, _queues
                )
                if success or failure:
                    log.info("OCR worker: %d ok / %d failed", success, failure)
            except Exception as exc:
                log.error("OCR worker error: %s", exc)
            _mark_worker_heartbeat("ocr")
            if await _sleep_or_stop():
                return

    async def _metadata_worker() -> None:
        while True:
            try:
                success, failure = await run_metadata_batch(
                    _paperless_client,
                    _config,
                    _queues,
                    custom_field_id,
                    ai_summary_field_id,
                    ai_result_field_id,
                )
                if success or failure:
                    log.info("Metadata worker: %d ok / %d failed", success, failure)
            except Exception as exc:
                log.error("Metadata worker error: %s", exc)
            _mark_worker_heartbeat("metadata")
            if await _sleep_or_stop():
                return

    async def _embed_worker() -> None:
        while True:
            try:
                success, failure = await run_embed_batch(
                    _paperless_client,
                    _config,
                    _queues,
                    store,
                    embedder,
                )
                if success or failure:
                    log.info("Embed worker: %d ok / %d failed", success, failure)
            except Exception as exc:
                log.error("Embed worker error: %s", exc)
            _mark_worker_heartbeat("embed")
            if await _sleep_or_stop():
                return

    async def _refresh_worker() -> None:
        while True:
            try:
                success, failure = await run_refresh_batch(
                    _paperless_client,
                    _config,
                    _queues,
                    store,
                )
                if success or failure:
                    log.info("Refresh worker: %d ok / %d failed", success, failure)
            except Exception as exc:
                log.error("Refresh worker error: %s", exc)
            _mark_worker_heartbeat("refresh")
            if await _sleep_or_stop():
                return

    now = time.time()
    _worker_heartbeats = {"ocr": now, "metadata": now, "embed": now, "refresh": now}
    _worker_tasks = [
        asyncio.create_task(_ocr_worker(), name="ocr-worker"),
        asyncio.create_task(_metadata_worker(), name="metadata-worker"),
        asyncio.create_task(_embed_worker(), name="embed-worker"),
        asyncio.create_task(_refresh_worker(), name="refresh-worker"),
    ]
    _worker_ready = True
    log.info("Copilot service ready with embedded worker runtime")
    if _local_search_warm_on_startup:
        _schedule_local_search_warmup("startup")
    yield

    if _local_search_warmup_task is not None:
        _local_search_warmup_task.cancel()
        await asyncio.gather(_local_search_warmup_task, return_exceptions=True)
    request_shutdown()
    if _worker_tasks:
        try:
            await asyncio.wait_for(
                asyncio.gather(*_worker_tasks, return_exceptions=True),
                timeout=30,
            )
        except asyncio.TimeoutError:
            for task in _worker_tasks:
                task.cancel()
            await asyncio.gather(*_worker_tasks, return_exceptions=True)
    if embedder is not None:
        await embedder.aclose()
    await close_model_probe_session()
    if store is not None:
        await store.aclose()
    _qdrant_store = None
    if _lazy_embedder is not None:
        await _lazy_embedder.aclose()
    if _queues is not None:
        await _queues.close()
    if _paperless_client is not None:
        await _paperless_client.aclose()
    _chat_copilot = None
    _chat_store = None
    _config = None
    _worker_tasks = []
    _worker_ready = False
    _worker_setup_error = None


app = FastAPI(lifespan=lifespan)


def _get_lazy_embedder() -> ProcessLocalSearchEmbedder:
    global _lazy_embedder
    if _lazy_embedder is None:
        from paperless_ai.search.local_search_process import ProcessLocalSearchEmbedder

        _lazy_embedder = ProcessLocalSearchEmbedder(
            idle_timeout_seconds=_local_search_idle_timeout_seconds,
            start_method=_local_search_start_method,
        )
        log.info(
            "Local search process initialized (model=%s)",
            ProcessLocalSearchEmbedder.LOCAL_RERANKER_MODEL_NAME,
        )
    return _lazy_embedder


def _get_chat_copilot() -> ChatCopilot:
    global _chat_copilot
    if _config is None or _paperless_client is None:
        raise RuntimeError("Paperless chat is not configured")
    if _chat_copilot is None:
        from paperless_ai.search.chat_agent import ChatCopilot

        _chat_copilot = ChatCopilot(
            _config,
            _paperless_client,
            _get_lazy_embedder(),
            _qdrant_url,
            qdrant_client=(
                _qdrant_store._client if _qdrant_store is not None else None
            ),
        )
        log.info("Chat copilot initialized")
    return _chat_copilot


def _schedule_local_search_warmup(reason: str) -> None:
    global _local_search_warmup_task
    embedder = _get_lazy_embedder()
    if _local_search_warmup_task is not None and not _local_search_warmup_task.done():
        return

    async def _run() -> None:
        try:
            log.info("Scheduling local search warmup (%s)", reason)
            await embedder.warmup()
            log.info("Local search warmup complete (%s)", reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("Local search warmup failed (%s): %s", reason, exc)

    _local_search_warmup_task = asyncio.create_task(_run())


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
      - Dense: Local BGE-M3 query embedding in the process-backed search worker → Qdrant cosine search
      - Keyword: Paperless full-text API
      - Merge: Reciprocal Rank Fusion (RRF) to combine incompatible score scales
      - Rerank: local bge-reranker-v2-m3 reorders fused candidates

    Returns doc_ids in final rank order (reranker score, descending).
    Gracefully degrades to dense-only search if Paperless is unavailable.
    """
    if _qdrant_store is not None and not await _qdrant_store.has_any_points():
        return JSONResponse(content=[])

    try:
        embedder = _get_lazy_embedder()
        filters = SearchFilters(
            correspondent=correspondent,
            document_type=document_type,
            storage_path=storage_path,
            tags=tags,
            year=year,
        )
        try:
            fused_ids, _chunk_map = await asyncio.wait_for(
                hybrid_retrieve(
                    embedder=embedder,
                    qdrant_url=_qdrant_url,
                    query=q,
                    client=_paperless_client,
                    filters=filters,
                    dense_k=K,
                    rerank_candidates=max(N, limit),
                    rrf_k=RRF_K,
                    qdrant_client=(
                        _qdrant_store._client if _qdrant_store is not None else None
                    ),
                ),
                timeout=_search_request_timeout_seconds,
            )
        except Exception as exc:
            log.warning(
                "Search: hybrid retrieval failed, returning empty results (%s: %s)",
                type(exc).__name__,
                exc,
            )
            return JSONResponse(content=[])

        return JSONResponse(content=fused_ids[:limit])

    except SystemExit, KeyboardInterrupt, GeneratorExit:
        raise
    except BaseException as exc:
        log.warning(
            "Search endpoint error (%s: %s) — returning empty results",
            type(exc).__name__,
            exc,
        )
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
    items: list[dict] = []
    for doc_id, flags in sorted(
        source_flags.items(),
        key=lambda item: (not item[1].get("inspected", False), item[0]),
    ):
        try:
            metadata = (
                await _paperless_client.get_document_chat_metadata(doc_id)
                if _paperless_client is not None
                else None
            )
        except Exception as exc:
            if not _is_unavailable_chat_source_error(exc):
                raise
            log.info("Could not refresh chat source %d: %s", doc_id, exc)
            metadata = None
        items.append(
            {
                **(metadata or {"id": doc_id, "title": f"Document {doc_id}"}),
                "matched": bool(flags.get("matched")),
                "inspected": bool(flags.get("inspected")),
                "available": metadata is not None,
                "detail_url": _document_detail_url(doc_id),
                "thumb_url": _document_thumb_url(doc_id),
                "preview_url": _document_preview_url(doc_id),
            }
        )
    return items


async def _restore_chat_sources(sources: list[dict]) -> list[dict]:
    """Refresh persisted source cards while retaining deleted-document snapshots."""
    items: list[dict] = []
    for source in sources:
        try:
            metadata = (
                await _paperless_client.get_document_chat_metadata(source["id"])
                if _paperless_client is not None
                else None
            )
        except Exception as exc:
            if not _is_unavailable_chat_source_error(exc):
                raise
            log.info("Could not refresh chat source %s: %s", source["id"], exc)
            metadata = None
        item = {**source, **(metadata or {}), "available": metadata is not None}
        item["detail_url"] = _document_detail_url(source["id"])
        item["thumb_url"] = _document_thumb_url(source["id"])
        item["preview_url"] = _document_preview_url(source["id"])
        items.append(item)
    return items


def _chat_owner(headers) -> str | None:
    """Return the reverse-proxy user identity, or the single-user owner key."""
    value = headers.get(os.environ.get("CHAT_OWNER_HEADER", "Remote-User"))
    return value.strip() if value and value.strip() else None


def _chat_title(payload: dict, *, required: bool = False) -> str | None:
    """Validate a conversation title, optional when creating a conversation."""
    title = payload.get("title")
    if title is None:
        if required:
            raise HTTPException(
                status_code=422, detail="title must be 1 to 120 characters"
            )
        return None
    if not isinstance(title, str) or not (title := title.strip()) or len(title) > 120:
        raise HTTPException(status_code=422, detail="title must be 1 to 120 characters")
    return title


def _chat_conversation_id(value: str) -> str:
    """Validate a conversation ID before passing it to the store."""
    try:
        return str(uuid.UUID(value))
    except (AttributeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="conversation_id must be a UUID"
        ) from exc


def _require_chat_store() -> ChatStore:
    """Return the initialized conversation store."""
    if _chat_store is None:
        raise HTTPException(status_code=503, detail="Chat history is unavailable")
    return _chat_store


@app.post("/conversations")
async def create_conversation(
    request: Request, payload: dict | None = None
) -> JSONResponse:
    """Create a durable conversation for the current user."""
    conversation = await _require_chat_store().create_conversation(
        _chat_owner(request.headers), _chat_title(payload or {})
    )
    return JSONResponse(content=conversation, status_code=201)


@app.get("/conversations")
async def list_conversations(request: Request) -> JSONResponse:
    """List the current user's durable conversations."""
    return JSONResponse(
        content={
            "items": await _require_chat_store().list_conversations(
                _chat_owner(request.headers)
            )
        }
    )


@app.get("/conversations/{conversation_id}")
async def load_conversation(conversation_id: str, request: Request) -> JSONResponse:
    """Load a conversation and refresh its source-card metadata."""
    conversation_id = _chat_conversation_id(conversation_id)
    conversation = await _require_chat_store().load_conversation(
        conversation_id, _chat_owner(request.headers)
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    for message in conversation["messages"]:
        message["sources"] = await _restore_chat_sources(message["sources"])
    return JSONResponse(content=conversation)


@app.patch("/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str, request: Request, payload: dict
) -> JSONResponse:
    """Rename one owned conversation."""
    conversation_id = _chat_conversation_id(conversation_id)
    conversation = await _require_chat_store().rename_conversation(
        conversation_id,
        _chat_owner(request.headers),
        _chat_title(payload, required=True),
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return JSONResponse(content=conversation)


@app.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request) -> Response:
    """Delete one owned conversation and its source references."""
    conversation_id = _chat_conversation_id(conversation_id)
    deleted = await _require_chat_store().delete_conversation(
        conversation_id, _chat_owner(request.headers)
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


@app.get("/chat")
async def chat_ui() -> HTMLResponse:
    """Serve the browser UI for the Paperless copilot."""
    if _local_search_warm_on_chat_load:
        _schedule_local_search_warmup("chat-ui")
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
    :root { color-scheme: light; --chat-bg:#f8faf9; --chat-panel:#fff; --chat-muted:#64748b; --chat-border:#dce3e8; --chat-primary:#17541f; --chat-primary-soft:#e4f0e6; --chat-radius:12px; }
    * { box-sizing: border-box; }
    html, body { width:100%; min-height:100%; margin:0; }
    body { overflow:hidden; background:var(--chat-bg); color:#111827; font-family:var(--bs-body-font-family,system-ui,sans-serif); }
    button, a, textarea { font:inherit; }
    :focus-visible { outline:3px solid #18752b!important; outline-offset:2px; }
    .paperless-topbar { min-height:52px; height:52px; background:var(--chat-primary); color:#fff; }
    .paperless-topbar-inner { height:100%; display:flex; align-items:center; gap:12px; padding:0 20px; }
    .brand-link { color:inherit; text-decoration:none; font-weight:700; }
    .brand-divider { height:24px; border-left:1px solid #ffffff80; }
    .topbar-title { font-weight:600; }
    .topbar-spacer { flex:1; }
    .connection-status { font-size:.875rem; }
    .topbar-button { min-width:44px; min-height:44px; border:0; background:transparent; color:inherit; border-radius:8px; }
    .chat-shell { height:calc(100dvh - 52px); height:calc(100vh - 52px); }
    @supports (height: 100dvh) { .chat-shell { height:calc(100dvh - 52px); } }
    .chat-layout { height:100%; display:grid; grid-template-columns:300px minmax(560px,1fr) minmax(440px,31vw); align-items:stretch; }
    .history-column, .chat-column, .preview-column { min-width:0; min-height:0; }
    .history-column { display:flex; flex-direction:column; background:var(--chat-bg); border-right:1px solid var(--chat-border); }
    .history-panel { display:flex; flex:1; flex-direction:column; min-height:0; overflow:hidden; }
    .history-header { display:flex; align-items:center; justify-content:space-between; gap:8px; padding:16px 12px; }
    .history-header strong { font-size:1rem; }
    #new-conversation { width:100%; min-height:46px; }
    .history-header { flex-wrap:wrap; }
    .history-list { display:grid; align-content:start; gap:12px; flex:1; min-height:0; overflow:auto; padding:8px 12px; }
    .history-group { display:grid; gap:4px; }
    .history-group h3 { margin:0; padding:4px 10px; color:var(--chat-muted); font-size:.75rem; font-weight:600; }
    .history-item { display:flex; width:100%; min-width:0; min-height:52px; flex-direction:column; justify-content:center; gap:2px; border:0; border-radius:8px; background:transparent; color:inherit; padding:7px 10px; text-align:left; overflow:hidden; }
    .history-item-title { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .history-item time { color:var(--chat-muted); font-size:.75rem; }
    .history-item:hover, .history-item.active { background:var(--chat-primary-soft); color:var(--chat-primary); }
    .history-empty { color:var(--chat-muted); font-size:.875rem; padding:12px; }
    .history-actions { display:flex; gap:8px; padding:12px; border-top:1px solid var(--chat-border); }
    .history-actions button { flex:1; min-height:44px; }
    .history-status { padding:0 12px; color:#842029; font-size:.875rem; }
    .history-status:empty { display:none; }
    .history-feedback { padding:0 12px; color:var(--chat-muted); font-size:.875rem; }
    .history-feedback:empty { display:none; }
    .archive-footer { display:grid; gap:6px; padding:12px 16px; color:var(--chat-muted); font-size:.875rem; border-top:1px solid var(--chat-border); }
    .archive-footer-heading { color:#111827; font-weight:600; }
    .archive-footer button { justify-self:start; min-height:36px; }
    .chat-column { display:flex; flex-direction:column; min-height:0; background:var(--chat-bg); }
    .chat-heading { padding:30px 20px 16px; }
    .chat-heading h1 { margin:0; font-size:2.125rem; line-height:1.2; font-weight:700; }
    .chat-heading p { margin:8px 0 0; color:var(--chat-muted); }
    .socket-banner { display:none; margin:0 20px 12px; }
    .socket-banner.active { display:block; }
    .chat-panel { display:flex; flex:1; flex-direction:column; min-height:0; }
    .conversation { flex:1; min-height:0; overflow:auto; display:grid; align-content:start; gap:16px; padding:12px 20px; scroll-behavior:smooth; }
    .empty-chat-invitation { margin:auto; max-width:34rem; padding:24px; color:var(--chat-muted); text-align:center; }
    .jump-to-latest { position:absolute; right:28px; bottom:112px; z-index:2; }
    .chat-panel { position:relative; }
    .message-time { justify-self:end; color:var(--chat-muted); font-size:.75rem; }
    .turn-time { color:var(--chat-muted); font-size:.75rem; }
    .turn-status { color:var(--chat-muted); font-size:.9rem; }
    .tool-card { padding:10px 12px; }
    .tool-card-header { display:flex; justify-content:space-between; gap:12px; }
    .tool-card-status { color:var(--chat-muted); }
    .tool-card details { margin-top:6px; }
    .tools-panel h3 { margin:0; padding:10px 12px 0; color:var(--chat-primary); font-size:.95rem; }
    .markdown-fallback { margin-top:8px; color:var(--chat-muted); font-size:.85rem; }
    .recovery-panel { display:none; margin:0 20px 8px; padding:12px; border:1px solid #f0c36d; border-radius:8px; background:#fff8e6; }
    .recovery-panel.active { display:grid; gap:8px; }
    .composer { display:grid; grid-template-columns:minmax(0,1fr) 44px; align-items:end; gap:12px; margin:8px 20px 16px; padding:8px; border:1px solid var(--chat-border); border-radius:16px; background:#fff; box-shadow:0 2px 8px #1118270d; padding-bottom:calc(8px + env(safe-area-inset-bottom)); }
    .composer textarea { width:100%; min-height:44px; max-height:35vh; resize:none; border:0; border-radius:999px; padding:11px 14px; line-height:1.5; overflow-y:auto; }
    .composer-actions .btn { width:44px; height:44px; padding:0; border-radius:50%; font-size:0; }
    .composer-actions .btn::after { content:'↑'; font-size:1.4rem; }
    .preview-column { min-width:0; background:#fff; border-left:1px solid var(--chat-border); }
    .preview-panel { height:100%; min-height:0; display:flex; flex-direction:column; overflow:auto; background:#fff; }
    .preview-placeholder { margin:auto; padding:24px; max-width:32ch; color:var(--chat-muted); text-align:center; }
    .preview-header { display:none; flex:none; align-items:flex-start; justify-content:space-between; gap:12px; padding:20px; border-bottom:1px solid var(--chat-border); }
    .preview-header.active { display:flex; }
    .preview-header h2 { font-size:1.25rem; line-height:1.3; }
    .preview-metadata { flex:none; padding:16px 20px; }
    .preview-metadata dl { display:grid; grid-template-columns:minmax(100px,.7fr) minmax(0,1.3fr); gap:8px 12px; margin:0; font-size:.875rem; }
    .preview-metadata dt { color:var(--chat-muted); font-weight:500; }
    .preview-metadata dd { min-width:0; margin:0; overflow-wrap:anywhere; }
    .preview-tags { display:flex; flex-wrap:wrap; gap:4px; }
    .preview-frame-wrap { display:none; min-height:320px; height:50vh; flex:none; padding:12px; background:#eef1ef; }
    .preview-frame-wrap.active { display:block; }
    .preview-state { margin:0 0 8px; color:var(--chat-muted); font-size:.875rem; }
    .preview-frame { width:100%; height:calc(100% - 28px); min-height:280px; border:0; border-radius:8px; background:#fff; box-shadow:0 2px 8px #11182712; }
    .preview-close { min-width:44px; min-height:44px; }
    .drawer-backdrop { display:none; }
    .skip-link { position:fixed; z-index:20; top:8px; left:8px; transform:translateY(-150%); padding:10px; background:#fff; color:#111827; }
    .skip-link:focus { transform:none; }
    dialog { width:min(420px,calc(100vw - 32px)); border:1px solid var(--chat-border); border-radius:12px; padding:24px; color:#111827; }
    dialog::backdrop { background:#11182780; }
    dialog button { min-height:44px; }
    .bubble { max-width:min(78ch,100%); padding:14px 16px; border-radius:12px; white-space:pre-wrap; box-shadow:0 2px 8px #0000000d; }
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
      white-space: normal;
    }
    .bubble.assistant.is-pending {
      color: var(--chat-muted);
    }
    .markdown-body > :first-child {
      margin-top: 0;
    }
    .markdown-body > :last-child {
      margin-bottom: 0;
    }
    .markdown-body p,
    .markdown-body ul,
    .markdown-body ol,
    .markdown-body pre,
    .markdown-body table,
    .markdown-body blockquote {
      margin: 0 0 0.85rem;
    }
    .markdown-body ul,
    .markdown-body ol {
      padding-left: 1.35rem;
    }
    .markdown-body li + li {
      margin-top: 0.25rem;
    }
    .markdown-body code {
      padding: 0.1rem 0.35rem;
      border-radius: 0.35rem;
      background: #f3f5f3;
      font-size: 0.92em;
    }
    .markdown-body pre {
      padding: 0.85rem 0.95rem;
      border-radius: 0.7rem;
      background: #f8f9fa;
      border: 1px solid var(--chat-border);
      overflow-x: auto;
    }
    .markdown-body pre code {
      padding: 0;
      background: transparent;
      border-radius: 0;
    }
    .markdown-body blockquote {
      padding-left: 0.9rem;
      border-left: 3px solid #cfe0d1;
      color: var(--chat-muted);
    }
    .markdown-body table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }
    .markdown-body th,
    .markdown-body td {
      padding: 0.45rem 0.55rem;
      border: 1px solid var(--chat-border);
      vertical-align: top;
    }
    .markdown-body th {
      background: #f8f9fa;
      font-weight: 600;
    }
    .markdown-body a {
      color: var(--chat-primary);
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
    .source-list { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); gap:12px; padding:0 0.95rem 0.95rem; }
    .source-card {
      display: grid;
      grid-template-columns:56px minmax(0,1fr);
      gap:8px;
      border:1px solid var(--chat-border);
      border-radius:8px;
      background:#fff;
      padding:10px;
      min-width:0;
    }
    .source-card.selected { border:2px solid #18752b; padding:9px; }
    .source-thumb {
      width:56px;
      height:76px;
      object-fit:contain;
      background:#edf1ed;
      border-radius:4px;
    }
    button.source-thumb { padding:0; border:0; background:#edf1ed; cursor:pointer; }
    .source-thumb-placeholder { display:grid; place-items:center; width:56px; height:76px; border-radius:4px; background:#edf1ed; color:var(--chat-muted); font-size:1.4rem; }
    .source-content {
      display:grid;
      gap:6px;
      min-width:0;
    }
    .source-title {
      margin:0;
      font-size:.9rem;
      line-height:1.3;
      font-weight:600;
    }
    .source-select { max-width:100%; padding:0; border:0; background:none; color:inherit; font-weight:inherit; text-align:left; overflow-wrap:anywhere; }
    .source-title a { color:inherit; text-decoration:none; overflow-wrap:anywhere;
    }
    .source-title a:hover {
      color: var(--chat-primary);
    }
    .source-badges,.source-tags,.source-meta,.source-actions { display:flex; flex-wrap:wrap; align-items:center; gap:4px; }
    .source-actions { grid-column:1/-1; }
    .source-badge,
    .source-meta span {
      display:inline-flex;
      align-items: center;
      padding:2px 6px;
      border-radius:999px;
      font-size:.7rem;
      border:1px solid var(--chat-border);
      background:#f8f9fa;
      color:var(--bs-secondary-text-emphasis);
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
    @media (min-width:1440px) { .chat-layout.inspector-closed { grid-template-columns:300px minmax(0,1fr) 0; } .chat-layout.inspector-closed .preview-column { display:none; } }
    @media (min-width:1200px) and (max-width:1439px) { .chat-layout { grid-template-columns:240px minmax(0,1fr); } .preview-column { position:fixed; z-index:12; inset:52px 0 0 auto; width:min(480px,100vw); transform:translateX(100%); transition:transform .18s ease; box-shadow:-8px 0 24px #1118271a; } .chat-layout[data-inspector-open="true"] .preview-column { transform:none; } }
    @media (max-width:1199px) { .chat-layout { grid-template-columns:minmax(0,1fr); } .history-column,.preview-column { position:fixed; z-index:12; top:52px; bottom:0; width:min(300px,calc(100vw - 32px)); background:#fff; box-shadow:8px 0 24px #1118271a; transition:transform .18s ease; } .history-column { left:0; transform:translateX(-105%); } .preview-column { right:0; width:min(480px,100vw); transform:translateX(105%); box-shadow:-8px 0 24px #1118271a; } .chat-layout[data-history-open="true"] .history-column,.chat-layout[data-inspector-open="true"] .preview-column { transform:none; } .drawer-backdrop.active { display:block; position:fixed; z-index:11; inset:52px 0 0; border:0; background:#11182766; } .topbar-button { display:inline-flex; align-items:center; justify-content:center; } }
    @media (min-width:1200px) { #history-toggle { display:none; } }
    @media (max-width:767px) { .paperless-topbar-inner { padding:0 8px; gap:8px; } .brand-label { max-width:108px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; } .connection-status { font-size:.75rem; } .chat-heading { padding:22px 12px 12px; } .chat-heading h1 { font-size:26px; } .conversation { padding:12px; } .composer { margin:8px 12px 12px; } .source-list { grid-template-columns:1fr; } .preview-column { width:100vw!important; } .preview-frame-wrap { height:45vh; min-height:280px; } }
    @media (max-width:400px) { .paperless-topbar-inner { gap:4px; padding-inline:4px; } .brand-divider { display:none; } .brand-label { max-width:42px; } .topbar-title { font-size:.8rem; white-space:nowrap; } .connection-status { max-width:64px; overflow-wrap:anywhere; font-size:.68rem; } .topbar-button { min-width:44px; padding:0 4px; font-size:.75rem; } }
    @media (prefers-reduced-motion:reduce) { *,*::before,*::after { scroll-behavior:auto!important; transition:none!important; animation:none!important; } }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">Skip to conversation</a>
  <header class="paperless-topbar">
    <div class="paperless-topbar-inner">
      <button id="history-toggle" class="topbar-button" type="button" aria-label="Open conversation history" aria-expanded="false" aria-controls="history-panel">☰</button>
      <a class="brand-link" href="/"><span aria-hidden="true">▤</span> <span class="brand-label">paperless-ngx</span></a>
      <span class="brand-divider" aria-hidden="true"></span>
      <span class="topbar-title">AI Copilot</span>
      <span class="topbar-spacer"></span>
      <span id="connection-status" class="connection-status" role="status" aria-live="polite">Connecting…</span>
      <button id="help-button" class="topbar-button" type="button" aria-haspopup="dialog">Help</button>
    </div>
  </header>
  <main id="main-content" class="chat-shell" tabindex="-1">
    <div id="chat-layout" class="chat-layout">
      <aside id="history-panel" class="history-column" aria-label="Conversation history">
        <nav class="history-panel" aria-labelledby="history-heading">
          <div class="history-header">
            <strong id="history-heading">Conversations</strong>
            <button id="new-conversation" type="button" class="btn btn-outline-primary">New chat</button>
          </div>
          <div id="history-list" class="history-list"></div>
          <div id="history-feedback" class="history-feedback" role="status" aria-live="polite"></div>
          <div id="history-status" class="history-status" role="alert" aria-live="assertive"></div>
          <div class="history-actions">
            <button id="rename-conversation" type="button" class="btn btn-sm btn-outline-secondary">Rename</button>
            <button id="delete-conversation" type="button" class="btn btn-sm btn-outline-danger">Delete</button>
          </div>
          <footer class="archive-footer" aria-label="Archive status">
            <span class="archive-footer-heading">Archive status</span>
            <span id="archive-status" role="status" aria-live="polite">Loading status…</span>
            <button id="history-refresh" type="button" class="btn btn-sm btn-outline-secondary">Refresh</button>
          </footer>
        </nav>
      </aside>
      <section class="chat-column" aria-labelledby="page-heading">
        <div class="chat-heading"><h1 id="page-heading" tabindex="-1">Ask your archive</h1><p>Search, inspect, and understand your Paperless documents.</p></div>
        <div id="socket-banner" class="socket-banner alert alert-warning mb-0" role="alert" aria-live="assertive"></div>
        <section class="chat-panel" aria-label="Conversation">
          <div id="conversation" class="conversation" aria-label="Conversation transcript"><div id="empty-chat-invitation" class="empty-chat-invitation">Ask a question to search and understand your archive.</div></div>
          <button id="jump-to-latest" type="button" class="jump-to-latest btn btn-sm btn-light" hidden>Jump to latest</button>
          <div id="chat-announcements" class="visually-hidden" role="status" aria-live="polite"></div>
          <div id="recovery-panel" class="recovery-panel" role="status" aria-live="polite"></div>
          <form id="chat-form" class="composer">
            <label class="visually-hidden" for="prompt">Message your archive</label>
            <textarea id="prompt" placeholder="Ask about invoices from 2024, documents from a correspondent, or the contents of a specific receipt..." aria-describedby="composer-help"></textarea>
            <span id="composer-help" class="visually-hidden">Press Enter to send. Press Shift and Enter for a new line. Control and Enter or Command and Enter also send.</span>
            <div class="composer-actions">
              <button id="send-button" type="submit" class="btn btn-primary px-4">Send</button>
            </div>
          </form>
        </section>
      </section>

      <aside id="preview-panel" class="preview-column" aria-label="Document inspector" aria-modal="false">
        <section class="preview-panel" aria-labelledby="preview-title">
          <div id="preview-placeholder" class="preview-placeholder">
            <h2 class="h5">Document inspector</h2>
            <p>Select a source to inspect it here. Use “Open in Paperless” for the full document view.</p>
          </div>
          <div id="preview-header" class="preview-header">
            <div>
              <h2 id="preview-title" class="h6 mb-1">Document preview</h2>
              <p id="preview-subtitle" class="text-muted small mb-0"></p>
            </div>
            <a id="preview-open-link" class="btn btn-sm btn-outline-secondary" target="_blank" rel="noopener noreferrer">Open in Paperless</a>
            <button id="preview-close" class="preview-close btn btn-outline-secondary" type="button" aria-label="Close document inspector">Close</button>
          </div>
          <div id="preview-metadata" class="preview-metadata" hidden><dl></dl></div>
          <div id="preview-frame-wrap" class="preview-frame-wrap">
            <p id="preview-state" class="preview-state" role="status" aria-live="polite"></p>
            <iframe id="preview-frame" class="preview-frame" title="Document preview"></iframe>
            <p id="preview-fallback" class="preview-state" hidden>Preview not displaying? <a id="preview-fallback-link" target="_blank" rel="noopener noreferrer">Open in Paperless</a></p>
          </div>
        </section>
      </aside>
      <button id="drawer-backdrop" class="drawer-backdrop" type="button" aria-label="Close drawer"></button>
    </div>
  </main>
  <dialog id="help-dialog" aria-labelledby="help-title">
    <h2 id="help-title">Using AI Copilot</h2>
    <p>Ask a question about your Paperless archive. Press Enter to send or Shift+Enter for a new line. Control+Enter or Command+Enter also sends.</p>
    <p>Each new connection starts without earlier conversation context. Saved conversations restore what you can read, but do not replay it to the assistant.</p>
    <form method="dialog"><button id="help-close" class="btn btn-primary">Close</button></form>
  </dialog>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
  <dialog id="rename-dialog" aria-labelledby="rename-title">
    <form id="rename-form">
      <h2 id="rename-title">Rename conversation</h2>
      <label for="rename-input">Conversation name</label>
      <input id="rename-input" class="form-control" maxlength="120" required>
      <div id="rename-error" role="alert"></div>
      <div class="d-flex justify-content-end gap-2 mt-3">
        <button id="rename-cancel" type="button" class="btn btn-outline-secondary">Cancel</button>
        <button id="rename-save" type="submit" class="btn btn-primary">Save</button>
      </div>
    </form>
  </dialog>
  <script>
    const conversation = document.getElementById("conversation");
    const emptyChatInvitation = document.getElementById("empty-chat-invitation");
    const jumpToLatest = document.getElementById("jump-to-latest");
    const recoveryPanel = document.getElementById("recovery-panel");
    const chatAnnouncements = document.getElementById("chat-announcements");
    const form = document.getElementById("chat-form");
    const prompt = document.getElementById("prompt");
    const sendButton = document.getElementById("send-button");
    const socketBanner = document.getElementById("socket-banner");
    const previewPlaceholder = document.getElementById("preview-placeholder");
    const previewHeader = document.getElementById("preview-header");
    const previewTitle = document.getElementById("preview-title");
    const previewSubtitle = document.getElementById("preview-subtitle");
    const previewOpenLink = document.getElementById("preview-open-link");
    const previewMetadata = document.getElementById("preview-metadata");
    const previewFrameWrap = document.getElementById("preview-frame-wrap");
    const previewState = document.getElementById("preview-state");
    const previewFallback = document.getElementById("preview-fallback");
    const previewFallbackLink = document.getElementById("preview-fallback-link");
    let previewFrame = document.getElementById("preview-frame");
    const historyList = document.getElementById("history-list");
    const historyStatus = document.getElementById("history-status");
    const historyFeedback = document.getElementById("history-feedback");
    const archiveStatus = document.getElementById("archive-status");
    const historyRefreshButton = document.getElementById("history-refresh");
    const chatLayout = document.getElementById("chat-layout");
    const historyColumn = document.getElementById("history-panel");
    const previewColumn = document.getElementById("preview-panel");
    const historyToggle = document.getElementById("history-toggle");
    const previewClose = document.getElementById("preview-close");
    const drawerBackdrop = document.getElementById("drawer-backdrop");
    const connectionStatus = document.getElementById("connection-status");
    const helpDialog = document.getElementById("help-dialog");
    const helpButton = document.getElementById("help-button");
    const newConversationButton = document.getElementById("new-conversation");
    const renameConversationButton = document.getElementById("rename-conversation");
    const deleteConversationButton = document.getElementById("delete-conversation");
    const renameDialog = document.getElementById("rename-dialog");
    const renameForm = document.getElementById("rename-form");
    const renameInput = document.getElementById("rename-input");
    const renameError = document.getElementById("rename-error");
    const renameSaveButton = document.getElementById("rename-save");
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const basePath = window.location.pathname.replace(/\/chat\/?$/, "");
    const wsPath = `${basePath}/ws/chat`.replace(/\/{2,}/g, "/");
    let socket = null;
    let socketGeneration = 0;
    let activeTurn = null;
    let recovery = null;
    let recoveryNeedsRefresh = false;
    let readerNearBottom = true;
    let markdownFallbackShown = false;
    let composing = false;
    const turns = new Map();
    let selectedConversationId = null;
    let conversations = [];
    let historyReady = false;
    let lifecycleLoading = false;
    let lifecycleMutation = false;
    let chatBusy = false;
    let historyRequest = 0;
    let selectionRequest = 0;
    let historyRetry = null;
    let drawerOpener = null;
    let openDrawerName = null;
    let selectedSource = null;
    let previewGeneration = 0;

    updateLifecycleControls();

    function drawerIsModal(name) {
      return name === "history"
        ? window.matchMedia("(max-width: 1199px)").matches
        : window.matchMedia("(max-width: 1439px)").matches;
    }

    function setDrawerInert(name) {
      const modal = Boolean(name && drawerIsModal(name));
      document.querySelector(".paperless-topbar").inert = modal;
      document.querySelector(".chat-column").inert = modal;
      historyColumn.inert = modal && name !== "history";
      previewColumn.inert = modal && name !== "inspector";
      const pane = name === "history" ? historyColumn : previewColumn;
      if (pane) pane.setAttribute("aria-modal", String(modal));
    }

    function openDrawer(name, opener = document.activeElement) {
      if ((name === "history" && window.matchMedia("(min-width:1200px)").matches)
          || (name === "inspector" && window.matchMedia("(min-width:1440px)").matches)) {
        if (name === "inspector") {
          drawerOpener = opener;
          chatLayout.classList.remove("inspector-closed");
        }
        return;
      }
      openDrawerName = name;
      drawerOpener = opener;
      historyToggle.setAttribute("aria-expanded", String(name === "history"));
      chatLayout.dataset.historyOpen = String(name === "history");
      chatLayout.dataset.inspectorOpen = String(name === "inspector");
      drawerBackdrop.classList.add("active");
      setDrawerInert(name);
      const pane = name === "history" ? historyColumn : previewColumn;
      pane.querySelector("button, a, [tabindex='0']")?.focus();
    }

    function closeDrawer({ restoreFocus = true } = {}) {
      const previous = openDrawerName;
      openDrawerName = null;
      historyToggle.setAttribute("aria-expanded", "false");
      delete chatLayout.dataset.historyOpen;
      delete chatLayout.dataset.inspectorOpen;
      drawerBackdrop.classList.remove("active");
      setDrawerInert(null);
      if (!previous && window.matchMedia("(min-width:1440px)").matches) {
        chatLayout.classList.add("inspector-closed");
      }
      if (restoreFocus && drawerOpener?.isConnected && !drawerOpener.closest("[inert]")) drawerOpener.focus();
      drawerOpener = null;
    }

    function syncDrawerMode() {
      if (openDrawerName && !drawerIsModal(openDrawerName)) {
        const name = openDrawerName;
        closeDrawer({ restoreFocus: false });
        if (name === "inspector") chatLayout.classList.remove("inspector-closed");
        document.getElementById("page-heading").focus({ preventScroll: true });
      } else if (openDrawerName) {
        setDrawerInert(openDrawerName);
      } else if (selectedSource && drawerIsModal("inspector")) {
        openDrawer("inspector", selectedSource.opener);
      }
    }

    historyToggle.addEventListener("click", (event) => openDrawer("history", event.currentTarget));
    previewClose.addEventListener("click", () => resetPreview({ restoreFocus: true }));
    drawerBackdrop.addEventListener("click", () => {
      if (openDrawerName === "inspector") resetPreview({ restoreFocus: true });
      else closeDrawer();
    });
    window.addEventListener("resize", syncDrawerMode);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && openDrawerName) {
        event.preventDefault();
        if (openDrawerName === "inspector") resetPreview({ restoreFocus: true });
        else closeDrawer();
      }
      if (event.key !== "Tab" || !openDrawerName || !drawerIsModal(openDrawerName)) return;
      const pane = openDrawerName === "history" ? historyColumn : previewColumn;
      const focusable = [...pane.querySelectorAll("button:not(:disabled),a[href],textarea:not(:disabled),[tabindex]:not([tabindex='-1'])")]
        .filter((element) => !element.hidden && element.getClientRects().length);
      const first = focusable[0];
      const last = focusable.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    });
    helpButton.addEventListener("click", () => helpDialog.showModal());
    helpDialog.addEventListener("close", () => helpButton.focus());

    function apiPath(path) {
      return `${basePath}${path}`.replace(/\/{2,}/g, "/");
    }

    async function api(path, options = {}) {
      const response = await fetch(apiPath(path), {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const error = new Error(body.detail || "Request failed.");
        error.status = response.status;
        throw error;
      }
      return response.status === 204 ? null : response.json();
    }

    function lifecycleBusy() {
      return lifecycleLoading || chatBusy;
    }

    function updateLifecycleControls() {
      const mutationDisabled = lifecycleBusy() || !historyReady;
      newConversationButton.disabled = mutationDisabled;
      renameConversationButton.hidden = !selectedConversationId;
      deleteConversationButton.hidden = !selectedConversationId;
      renameConversationButton.disabled = mutationDisabled || !selectedConversationId;
      deleteConversationButton.disabled = mutationDisabled || !selectedConversationId;
      historyList.querySelectorAll("button").forEach((button) => {
        button.disabled = chatBusy || lifecycleMutation || button.dataset.conversationId === selectedConversationId;
      });
      historyRefreshButton.disabled = lifecycleLoading;
      sendButton.disabled = !prompt.value.trim() || !historyReady || lifecycleLoading || chatBusy || recoveryNeedsRefresh || !socket || socket.readyState !== WebSocket.OPEN;
    }

    // Step 03 calls this hook when a turn starts and completes.
    function setChatBusy(busy) {
      chatBusy = Boolean(busy);
      historyFeedback.textContent = chatBusy ? "A response is in progress. Conversation controls are temporarily unavailable." : "";
      updateLifecycleControls();
    }

    function setLifecycleLoading(loading) {
      lifecycleLoading = loading;
      updateLifecycleControls();
    }

    function setLifecycleMutation(loading) {
      lifecycleMutation = loading;
      setLifecycleLoading(loading);
    }

    function setHistoryError(message, retry) {
      historyStatus.replaceChildren();
      historyFeedback.textContent = "";
      const text = document.createElement("span");
      text.textContent = message;
      historyStatus.appendChild(text);
      historyRetry = retry;
      if (retry) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-sm btn-outline-danger ms-2";
        button.textContent = "Retry";
        button.addEventListener("click", () => historyRetry?.());
        historyStatus.appendChild(button);
      }
    }

    function clearHistoryError() {
      historyStatus.replaceChildren();
      historyFeedback.textContent = "";
      historyRetry = null;
    }

    function resetPreview({ restoreFocus = false } = {}) {
      const opener = selectedSource?.opener;
      previewGeneration += 1;
      previewFrame.removeAttribute("src");
      previewFrame.replaceWith(previewFrame.cloneNode(false));
      previewFrame = document.getElementById("preview-frame");
      previewFrameWrap.classList.remove("active");
      previewHeader.classList.remove("active");
      previewMetadata.hidden = true;
      previewMetadata.querySelector("dl").replaceChildren();
      previewFallback.hidden = true;
      previewState.textContent = "";
      previewPlaceholder.hidden = false;
      previewOpenLink.removeAttribute("href");
      previewFallbackLink.removeAttribute("href");
      selectedSource = null;
      document.querySelectorAll(".source-card.selected").forEach((card) => card.classList.remove("selected"));
      document.querySelectorAll(".source-select[aria-pressed='true'], .source-preview[aria-pressed='true']")
        .forEach((button) => button.setAttribute("aria-pressed", "false"));
      chatLayout.classList.add("inspector-closed");
      if (openDrawerName === "inspector") closeDrawer({ restoreFocus: false });
      if (restoreFocus && opener?.isConnected) opener.focus();
    }

    function clearConversationView() {
      resetPreview();
      conversation.replaceChildren();
      emptyChatInvitation.hidden = false;
      conversation.appendChild(emptyChatInvitation);
      turns.clear();
      readerNearBottom = true;
      jumpToLatest.hidden = true;
    }

    function dateGroup(timestamp) {
      const date = new Date(timestamp);
      if (Number.isNaN(date.getTime())) return "Older";
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const day = new Date(date.getFullYear(), date.getMonth(), date.getDate());
      const ordinal = (value) => Date.UTC(value.getFullYear(), value.getMonth(), value.getDate()) / 86400000;
      const diff = ordinal(today) - ordinal(day);
      return diff === 0 ? "Today" : diff >= 1 && diff <= 7 ? "Previous 7 days" : "Older";
    }

    function formatHistoryTime(timestamp, group) {
      const date = new Date(timestamp);
      if (Number.isNaN(date.getTime())) return "Date unavailable";
      return group === "Today"
        ? new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date)
        : new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric" }).format(date);
    }

    function renderConversationList() {
      historyList.replaceChildren();
      if (!conversations.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent = historyReady ? "No conversations yet." : "Loading conversations…";
        historyList.appendChild(empty);
      }
      const groups = new Map([["Today", []], ["Previous 7 days", []], ["Older", []]]);
      conversations.forEach((item) => groups.get(dateGroup(item.updated_at)).push(item));
      groups.forEach((items, label) => {
        if (!items.length) return;
        const group = document.createElement("section");
        group.className = "history-group";
        const heading = document.createElement("h3");
        heading.textContent = label;
        group.appendChild(heading);
        items.forEach((item) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `history-item ${item.id === selectedConversationId ? "active" : ""}`;
          button.dataset.conversationId = item.id;
          button.title = item.title;
          button.setAttribute("aria-label", `${item.title}, updated ${new Date(item.updated_at).toLocaleString()}`);
          if (item.id === selectedConversationId) button.setAttribute("aria-current", "page");
          const title = document.createElement("span");
          title.className = "history-item-title";
          title.textContent = item.title;
          const time = document.createElement("time");
          time.dateTime = item.updated_at;
          time.textContent = formatHistoryTime(item.updated_at, label);
          time.title = new Date(item.updated_at).toLocaleString();
          button.append(title, time);
          button.addEventListener("click", () => selectConversation(item.id));
          group.appendChild(button);
        });
        historyList.appendChild(group);
      });
      updateLifecycleControls();
    }

    async function refreshConversations() {
      const request = ++historyRequest;
      const items = (await api("/conversations")).items;
      if (request !== historyRequest) return conversations;
      conversations = items;
      renderConversationList();
      return conversations;
    }

    async function createConversation() {
      if (lifecycleBusy()) return;
      if (selectedConversationId && conversation.childElementCount === 0 && !prompt.value.trim()) {
        prompt.focus();
        return;
      }
      if (prompt.value.trim() && !window.confirm("Discard the unsent draft and start a new conversation?")) return;
      setLifecycleMutation(true);
      clearHistoryError();
      try {
        const item = await api("/conversations", { method: "POST", body: "{}" });
        selectedConversationId = item.id;
        historyReady = true;
        clearConversationView();
        prompt.value = "";
        renderRecovery();
        if (openDrawerName === "history") closeDrawer({ restoreFocus: false });
        conversations = [item, ...conversations.filter((conversation) => conversation.id !== item.id)];
        renderConversationList();
        await refreshConversations().catch((error) => setHistoryError(`Conversation created, but history could not refresh: ${error.message}`, () => refreshConversations().then(clearHistoryError).catch((retryError) => setHistoryError(`Unable to refresh conversations: ${retryError.message}`, () => retryHistory()))));
        prompt.focus();
      } catch (error) {
        setHistoryError(`Unable to create a conversation: ${error.message}`, () => refreshConversations().then(clearHistoryError).catch((retryError) => setHistoryError(`Unable to refresh conversations: ${retryError.message}`, () => retryHistory())));
      } finally {
        setLifecycleMutation(false);
      }
    }

    async function selectConversation(conversationId, { discardDraft = false, force = false, preserveDraft = false } = {}) {
      if (!conversationId || chatBusy || lifecycleMutation || (conversationId === selectedConversationId && !force)) return;
      if (!discardDraft && prompt.value.trim() && !window.confirm("Discard the unsent draft and switch conversations?")) return;
      const request = ++selectionRequest;
      const draftRevision = promptRevision;
      setLifecycleLoading(true);
      clearHistoryError();
      historyFeedback.textContent = "Loading conversation…";
      try {
        const item = await api(`/conversations/${conversationId}`);
        if (request !== selectionRequest) return false;
        if (promptRevision !== draftRevision) {
          historyFeedback.textContent = "Draft changed while loading. Conversation switch canceled; the draft was kept.";
          return false;
        }
        selectedConversationId = item.id;
        historyReady = true;
        clearConversationView();
        if (openDrawerName === "history") closeDrawer({ restoreFocus: false });
        item.messages.forEach((message) => {
          if (message.role === "user") {
            addUserBubble(message.content, message.created_at);
            return;
          }
          createTurn(message.id, message.created_at);
          setAssistantMessage(message.id, message.content);
          (message.tool_activity || []).forEach((tool, position) => updateTool(message.id, tool, false, position));
          updateUsage(message.id, { available: Boolean(message.usage), model: message.model, ...(message.usage || {}) });
          renderSources(message.id, message.sources || []);
        });
        addContextResetNotice();
        if (!preserveDraft) prompt.value = "";
        renderConversationList();
        historyFeedback.textContent = "";
        renderRecovery();
        await refreshConversations().catch((error) => setHistoryError(`Conversation loaded, but history could not refresh: ${error.message}`, () => retryHistory()));
        return true;
      } catch (error) {
        if (request !== selectionRequest) return false;
        if (error.status === 404) {
          if (selectedConversationId === conversationId) {
            selectedConversationId = null;
            historyReady = false;
            clearConversationView();
          }
          historyFeedback.textContent = "";
          setHistoryError("This conversation is unavailable. Refresh history and choose another conversation.", () => retryHistory());
          await refreshConversations().catch(() => {});
        } else {
          historyFeedback.textContent = "";
          setHistoryError(`Unable to load this conversation: ${error.message}`, () => selectConversation(conversationId));
        }
        return false;
      } finally {
        if (request === selectionRequest) setLifecycleLoading(false);
      }
    }

    async function retryHistory() {
      try {
        await refreshConversations();
        if (selectedConversationId) await selectConversation(selectedConversationId, { discardDraft: true, force: true, preserveDraft: true });
        else if (conversations.length) await selectConversation(conversations[0].id);
        else {
          historyReady = true;
          updateLifecycleControls();
          await createConversation();
        }
      } catch (error) {
        setHistoryError(`Unable to refresh conversations: ${error.message}`, () => retryHistory());
      }
    }

    async function loadArchiveStatus() {
      archiveStatus.textContent = "Loading status…";
      try {
        const response = await fetch(apiPath("/health"));
        const body = await response.json();
        if (!body || !["ok", "degraded"].includes(body.status)) throw new Error("Invalid health response");
        archiveStatus.textContent = body.status === "ok" ? "Processing available" : "Processing degraded";
        if (body.pending && typeof body.pending === "object") {
          const pending = Object.values(body.pending).reduce((total, count) => total + (Number(count) || 0), 0);
          archiveStatus.textContent += ` · ${pending} pending tasks`;
        }
      } catch {
        archiveStatus.textContent = "Status unavailable";
      }
    }

    function scrollConversation(force = false) {
      if (!force && !readerNearBottom) {
        jumpToLatest.hidden = false;
        return;
      }
      conversation.scrollTop = conversation.scrollHeight;
      jumpToLatest.hidden = true;
      readerNearBottom = true;
    }

    conversation.addEventListener("scroll", () => {
      readerNearBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight <= 64;
      jumpToLatest.hidden = readerNearBottom;
    });
    jumpToLatest.addEventListener("click", () => scrollConversation(true));

    function autosizePrompt() {
      prompt.style.height = "auto";
      const maxHeight = Math.min(160, Math.max(88, window.innerHeight * 0.2));
      prompt.style.height = `${Math.min(Math.max(prompt.scrollHeight, sendButton.offsetHeight), maxHeight)}px`;
    }

    function setSocketBanner(kind, text, reconnect = false) {
      socketBanner.className = `socket-banner alert alert-${kind} mb-0 active`;
      socketBanner.replaceChildren(document.createTextNode(text));
      if (reconnect) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "btn btn-sm btn-outline-secondary ms-2";
        button.textContent = "Reconnect";
        button.addEventListener("click", reconnectSocket);
        socketBanner.appendChild(button);
      }
      connectionStatus.textContent = kind === "danger" ? "Connection error" : kind === "warning" ? "Disconnected" : "Connected";
    }

    function clearSocketBanner() {
      socketBanner.className = "socket-banner alert alert-warning mb-0";
      socketBanner.replaceChildren();
      connectionStatus.textContent = "Connected";
    }

    function renderMarkdown(text) {
      const source = String(text ?? "").trim() || "(no response)";
      if (!window.marked?.parse || !window.DOMPurify?.sanitize) return null;
      try {
        return window.DOMPurify.sanitize(window.marked.parse(source, { gfm: true, breaks: false }));
      } catch {
        return null;
      }
    }

    function refreshEmptyState() {
      emptyChatInvitation.hidden = conversation.querySelector(".bubble.user, .turn, .context-reset-notice") !== null;
    }

    function addUserBubble(content, createdAt = null) {
      const bubble = document.createElement("div");
      bubble.className = "bubble user";
      bubble.textContent = content;
      conversation.appendChild(bubble);
      const time = document.createElement("time");
      time.className = "message-time";
      const date = createdAt ? new Date(createdAt) : new Date();
      if (!Number.isNaN(date.getTime())) {
        time.dateTime = date.toISOString();
        time.textContent = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
        time.title = createdAt ? date.toLocaleString() : "Received in this browser";
        conversation.appendChild(time);
      }
      refreshEmptyState();
      scrollConversation(true);
      return bubble;
    }

    let promptRevision = 0;
    prompt.addEventListener("input", () => {
      promptRevision += 1;
      autosizePrompt();
    });
    window.addEventListener("load", autosizePrompt);
    window.addEventListener("resize", autosizePrompt);
    autosizePrompt();

    function createTurn(turnId, createdAt = null) {
      if (turns.has(turnId)) return turns.get(turnId);
      const turn = document.createElement("article");
      turn.className = "turn";
      turn.dataset.turnId = turnId;
      turn.setAttribute("aria-label", "Assistant response");

      const time = document.createElement("time");
      time.className = "turn-time";
      if (createdAt) {
        const date = new Date(createdAt);
        if (!Number.isNaN(date.getTime())) {
          time.dateTime = date.toISOString();
          time.textContent = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
        }
      }

      const status = document.createElement("div");
      status.className = "turn-status";
      status.textContent = "Waiting for the assistant…";
      const answer = document.createElement("div");
      answer.className = "bubble assistant is-pending";
      const timeline = document.createElement("div");
      timeline.className = "timeline";
      const usage = document.createElement("div");
      usage.className = "usage";
      const toolsPanel = document.createElement("section");
      toolsPanel.className = "tools-panel";
      const toolsSummary = document.createElement("h3");
      toolsSummary.textContent = "Activity";
      const toolList = document.createElement("div");
      toolList.className = "tool-list";
      toolsPanel.append(toolsSummary, toolList);
      const sourcesPanel = document.createElement("details");
      sourcesPanel.className = "sources-panel";
      const sourcesSummary = document.createElement("summary");
      sourcesSummary.textContent = "Sources";
      const sourceList = document.createElement("div");
      sourceList.className = "source-list";
      sourcesPanel.append(sourcesSummary, sourceList);
      toolsPanel.hidden = true;
      sourcesPanel.hidden = true;
      turn.append(time, status, timeline, toolsPanel, answer, sourcesPanel, usage);
      conversation.appendChild(turn);

      const state = { root: turn, answer, status, timeline, usage, toolsPanel, toolList, toolsSummary, sourcesPanel, sourceList, sourcesSummary, toolEntries: new Map() };
      turns.set(turnId, state);
      refreshEmptyState();
      scrollConversation();
      return state;
    }

    function addTimelineItem(turnId, text) {
      const turn = turns.get(turnId);
      if (!turn) return;
      turn.status.textContent = text;
      chatAnnouncements.textContent = text;
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
      const turn = turns.get(turnId);
      if (!turn) return;
      turn.usage.replaceChildren();
      if (payload.model) turn.usage.appendChild(renderUsageChip("Model", payload.model));
      if (payload.available !== true) {
        turn.usage.appendChild(renderUsageChip("Tokens", "unavailable"));
        return;
      }
      for (const [label, key] of [["Prompt", "prompt_tokens"], ["Completion", "completion_tokens"], ["Total", "total_tokens"]]) {
        turn.usage.appendChild(renderUsageChip(label, Number.isFinite(payload[key]) ? payload[key] : "unavailable"));
      }
    }

    function formatJson(value) {
      try { return JSON.stringify(value, null, 2); }
      catch { return String(value); }
    }

    function toolLabel(name) {
      return ({ get_available_metadata: "Check archive metadata", search_documents: "Search documents", read_document: "Read document" })[name] || name || "Unknown tool";
    }

    function getOrCreateToolCard(turnId, payload, position = 0) {
      const turn = turns.get(turnId);
      if (!turn) return null;
      turn.toolsPanel.hidden = false;
      const key = String(payload.tool_call_id || `position-${position}`);
      if (turn.toolEntries.has(key)) return turn.toolEntries.get(key);
      const card = document.createElement("div");
      card.className = "tool-card";
      const header = document.createElement("div");
      header.className = "tool-card-header";
      const label = document.createElement("strong");
      label.textContent = toolLabel(payload.name);
      const state = document.createElement("span");
      state.className = "tool-card-status";
      header.append(label, state);
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = "Details";
      const body = document.createElement("div");
      body.className = "tool-body";
      const meta = document.createElement("div");
      meta.className = "tool-meta";
      const preview = document.createElement("pre");
      preview.className = "tool-preview";
      const args = document.createElement("pre");
      args.className = "tool-arguments";
      args.textContent = formatJson(payload.arguments ?? {});
      body.append(meta, preview, args);
      details.append(summary, body);
      card.append(header, details);
      turn.toolList.appendChild(card);
      const tool = { card, details, meta, preview, args, label, state };
      turn.toolEntries.set(key, tool);
      turn.toolsSummary.textContent = `Activity (${turn.toolEntries.size})`;
      return tool;
    }

    function updateTool(turnId, payload, started, position = 0) {
      const tool = getOrCreateToolCard(turnId, payload, position);
      if (!tool) return;
      tool.label.textContent = toolLabel(payload.name);
      tool.args.textContent = formatJson(payload.arguments ?? {});
      if (started) {
        tool.state.textContent = "Running…";
        tool.meta.replaceChildren();
        tool.preview.textContent = "";
      } else {
        tool.state.textContent = payload.interrupted ? "Interrupted; outcome unknown" : payload.summary || "Complete";
        const pieces = [];
        if (payload.duration_ms != null) pieces.push(`Duration ${payload.duration_ms} ms`);
        tool.meta.replaceChildren(...pieces.map((text) => {
          const span = document.createElement("span");
          span.textContent = text;
          return span;
        }));
        tool.preview.textContent = payload.preview || "";
        tool.details.hidden = !payload.arguments && !payload.preview;
      }
      scrollConversation();
    }

    function interruptRunningTools(turnId) {
      const turn = turns.get(turnId);
      if (!turn) return;
      turn.toolEntries.forEach((tool) => {
        if (tool.state.textContent === "Running…") {
          tool.state.textContent = "Interrupted; outcome unknown";
          tool.card.dataset.interrupted = "true";
        }
      });
    }

    function getTurn(turnId) {
      return turns.get(turnId) || null;
    }

    function safeSourceText(value, fallback = "Not available") {
      return (typeof value === "string" || typeof value === "number") && String(value).trim() ? String(value) : fallback;
    }

    function sourceDocumentId(source) {
      const value = source?.id;
      if (typeof value !== "number" && !(typeof value === "string" && /^[1-9]\d*$/.test(value))) return null;
      const id = Number(value);
      return Number.isSafeInteger(id) && id > 0 ? id : null;
    }

    function sourcePath(id, kind) {
      return kind === "detail" ? `/documents/${id}/detail`
        : kind === "thumb" ? `/api/documents/${id}/thumb/`
          : `/api/documents/${id}/preview/`;
    }

    function formatDocumentDate(value) {
      if (typeof value !== "string" || !value) return "Not available";
      const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
      let date;
      if (dateOnly) {
        const [year, month, day] = dateOnly.slice(1).map(Number);
        date = new Date(0);
        date.setFullYear(year, month - 1, day);
        date.setHours(0, 0, 0, 0);
        if (date.getFullYear() !== year || date.getMonth() !== month - 1 || date.getDate() !== day) return "Not available";
      } else date = new Date(value);
      if (Number.isNaN(date.getTime())) return "Not available";
      return date.toLocaleDateString();
    }

    function renderSourceBadges(source) {
      const badges = [];
      if (source.available === false) badges.push(["Unavailable", ""]);
      if (source.matched) badges.push(["Matched", "match"]);
      if (source.inspected) badges.push(["Read in full", "read"]);
      return badges;
    }

    function updateSourceSelection() {
      document.querySelectorAll(".source-card").forEach((card) => {
        const selected = Boolean(selectedSource)
          && card.dataset.sourceKey === selectedSource.key;
        card.classList.toggle("selected", selected);
        card.querySelectorAll(".source-select, .source-preview").forEach((button) => {
          button.setAttribute("aria-pressed", String(selected));
        });
      });
    }

    function appendMetadataRow(list, label, value) {
      const term = document.createElement("dt");
      term.textContent = label;
      const description = document.createElement("dd");
      if (value instanceof Node) description.appendChild(value);
      else description.textContent = value;
      list.append(term, description);
    }

    function openPreview(source, turnId, opener = document.activeElement) {
      const id = sourceDocumentId(source);
      const isAvailable = source.available !== false && id !== null;
      const title = safeSourceText(source.title, id ? `Document ${id}` : "Document unavailable");
      const key = JSON.stringify([selectedConversationId, turnId, id ?? safeSourceText(source.id, "unknown")]);
      selectedSource = { key, opener };
      updateSourceSelection();
      previewPlaceholder.hidden = true;
      previewHeader.classList.add("active");
      previewMetadata.hidden = false;
      previewTitle.textContent = title;
      previewSubtitle.textContent = id ? `Document ${id}` : "Document ID unavailable";
      previewOpenLink.hidden = id === null;
      previewFallbackLink.hidden = id === null;
      if (id !== null) {
        previewOpenLink.href = sourcePath(id, "detail");
        previewFallbackLink.href = sourcePath(id, "detail");
      } else {
        previewOpenLink.removeAttribute("href");
        previewFallbackLink.removeAttribute("href");
      }

      const metadata = previewMetadata.querySelector("dl");
      metadata.replaceChildren();
      const tags = Array.isArray(source.tag_names)
        ? source.tag_names.filter((tag) => typeof tag === "string" && tag.trim())
        : [];
      const tagList = document.createElement("div");
      tagList.className = "preview-tags";
      if (tags.length) {
        tags.forEach((tag) => {
          const chip = document.createElement("span");
          chip.className = "source-badge";
          chip.textContent = tag;
          tagList.appendChild(chip);
        });
      } else tagList.textContent = "No tags";
      appendMetadataRow(metadata, "Correspondent", safeSourceText(source.correspondent_name));
      appendMetadataRow(metadata, "Document date", formatDocumentDate(source.created));
      appendMetadataRow(metadata, "Document type", safeSourceText(source.document_type_name));
      appendMetadataRow(metadata, "Tags", tagList);
      appendMetadataRow(metadata, "Added", formatDocumentDate(source.added));
      appendMetadataRow(metadata, "File name", safeSourceText(source.original_filename));
      appendMetadataRow(metadata, "Document ID", id === null ? "Not available" : String(id));
      appendMetadataRow(metadata, "Archive serial number", safeSourceText(source.archive_serial_number));
      appendMetadataRow(metadata, "Storage path", safeSourceText(source.storage_path_name));

      previewGeneration += 1;
      const generation = previewGeneration;
      previewFrame.removeAttribute("src");
      const nextFrame = previewFrame.cloneNode(false);
      previewFrame.replaceWith(nextFrame);
      previewFrame = nextFrame;
      previewFrame.title = `Preview of ${title}`;
      previewFrameWrap.classList.add("active");
      previewFallback.hidden = true;
      if (!isAvailable) {
        previewState.textContent = source.available === false
          ? "Document unavailable. Saved metadata is shown; preview and thumbnail are disabled."
          : "Preview unavailable because the document ID is invalid.";
        previewFallback.hidden = source.available === false || id === null;
      } else {
        previewState.textContent = "Loading preview…";
        previewFallback.hidden = false;
        nextFrame.addEventListener("load", () => {
          if (generation === previewGeneration && previewFrame === nextFrame) previewState.textContent = "";
        });
        nextFrame.addEventListener("error", () => {
          if (generation === previewGeneration && previewFrame === nextFrame) previewState.textContent = "Preview could not be loaded.";
        });
        nextFrame.src = sourcePath(id, "preview");
      }
      openDrawer("inspector", opener);
    }

    function renderSources(turnId, items) {
      const turn = getTurn(turnId);
      if (!turn) return;
      turn.sourcesPanel.hidden = items.length === 0;
      turn.sourcesPanel.open = true;
      const uniqueItems = [...new Map(items.map((source) => [String(sourceDocumentId(source) ?? source.id), source])).values()];
      turn.sourcesSummary.textContent = `Sources (${uniqueItems.length})`;
      turn.sourceList.replaceChildren();
      uniqueItems.forEach((source) => {
        const id = sourceDocumentId(source);
        const titleText = safeSourceText(source.title, id ? `Document ${id}` : "Document unavailable");
        const key = JSON.stringify([selectedConversationId, turnId, id ?? safeSourceText(source.id, "unknown")]);
        const card = document.createElement("article");
        card.className = "source-card";
        card.dataset.sourceKey = key;

        const selectButton = document.createElement("button");
        selectButton.type = "button";
        selectButton.className = "source-select source-thumb";
        selectButton.setAttribute("aria-label", `Inspect ${titleText}`);
        selectButton.setAttribute("aria-pressed", "false");
        const thumb = document.createElement("img");
        thumb.className = "source-thumb";
        thumb.loading = "lazy";
        thumb.alt = "";
        if (source.available !== false && id !== null) {
          thumb.src = sourcePath(id, "thumb");
          thumb.onerror = () => {
            const placeholder = document.createElement("span");
            placeholder.className = "source-thumb-placeholder";
            placeholder.setAttribute("aria-hidden", "true");
            placeholder.textContent = "▤";
            thumb.replaceWith(placeholder);
          };
        } else {
          const placeholder = document.createElement("span");
          placeholder.className = "source-thumb-placeholder";
          placeholder.setAttribute("aria-hidden", "true");
          placeholder.textContent = "▤";
          selectButton.appendChild(placeholder);
        }
        if (source.available !== false && id !== null) selectButton.appendChild(thumb);
        selectButton.addEventListener("click", () => openPreview(source, turnId, selectButton));

        const content = document.createElement("div");
        content.className = "source-content";

        const title = document.createElement("h3");
        title.className = "source-title";
        const titleButton = document.createElement("button");
        titleButton.type = "button";
        titleButton.className = "source-select";
        titleButton.textContent = titleText;
        titleButton.setAttribute("aria-pressed", "false");
        titleButton.addEventListener("click", () => openPreview(source, turnId, titleButton));
        title.appendChild(titleButton);

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
        const sourceTags = Array.isArray(source.tag_names)
          ? source.tag_names.filter((tag) => typeof tag === "string" && tag.trim()) : [];
        [formatDocumentDate(source.created), safeSourceText(source.correspondent_name, ""),
          ...sourceTags.slice(0, 2), sourceTags.length > 2 ? `+${sourceTags.length - 2} more` : "",
        ].filter((item) => item && item !== "Not available").forEach((item) => {
          const span = document.createElement("span");
          span.textContent = item;
          meta.appendChild(span);
        });

        const actions = document.createElement("div");
        actions.className = "source-actions";
        const previewButton = document.createElement("button");
        previewButton.type = "button";
        previewButton.className = "source-preview btn btn-sm btn-outline-secondary";
        previewButton.textContent = source.available === false || id === null ? "Preview unavailable" : "Preview document";
        previewButton.disabled = source.available === false || id === null;
        previewButton.setAttribute("aria-pressed", "false");
        previewButton.addEventListener("click", () => openPreview(source, turnId, previewButton));

        let openLink;
        if (id !== null) {
          openLink = document.createElement("a");
          openLink.className = "btn btn-sm btn-link px-0";
          openLink.href = sourcePath(id, "detail");
          openLink.target = "_blank";
          openLink.rel = "noopener noreferrer";
          openLink.textContent = "Open in Paperless";
        }

        const inspectButton = document.createElement("button");
        inspectButton.type = "button";
        inspectButton.className = "btn btn-sm btn-link px-0";
        inspectButton.textContent = "Inspect details";
        inspectButton.addEventListener("click", () => openPreview(source, turnId, inspectButton));

        actions.appendChild(previewButton);
        actions.appendChild(inspectButton);
        if (openLink) actions.appendChild(openLink);

        content.appendChild(title);
        content.appendChild(badges);
        content.appendChild(meta);
        card.appendChild(selectButton);
        card.appendChild(content);
        card.appendChild(actions);
        turn.sourceList.appendChild(card);
      });
      updateSourceSelection();
    }

    function setAssistantMessage(turnId, content) {
      const turn = turns.get(turnId);
      if (!turn) return;
      turn.answer.classList.remove("is-pending");
      const html = renderMarkdown(content);
      if (html === null) {
        turn.answer.textContent = String(content ?? "(no response)");
        if (!markdownFallbackShown) {
          const notice = document.createElement("p");
          notice.className = "markdown-fallback";
          notice.textContent = "Markdown formatting is unavailable; showing the answer as plain text.";
          turn.answer.appendChild(notice);
          markdownFallbackShown = true;
        }
      } else {
        turn.answer.innerHTML = `<div class="markdown-body">${html}</div>`;
      }
      turn.status.textContent = "Answer ready.";
      scrollConversation();
    }

    function setTurnError(turnId, content) {
      const turn = turns.get(turnId);
      if (!turn) return;
      turn.answer.classList.remove("is-pending");
      turn.answer.textContent = content;
      turn.answer.setAttribute("role", "alert");
      turn.answer.classList.add("border", "border-danger-subtle", "bg-danger-subtle");
      turn.status.textContent = "Response failed. The prompt was not saved.";
      scrollConversation();
    }

    function renderRecovery() {
      recoveryPanel.replaceChildren();
      recoveryPanel.classList.toggle("active", Boolean(recovery));
      if (!recovery) return;
      const message = document.createElement("div");
      message.textContent = recovery.failed
        ? "This prompt failed and was not saved. You can restore it to the composer and try again."
        : "The previous response may still be saving. Refresh history before resending; the request may already have completed.";
      const submitted = document.createElement("div");
      submitted.textContent = `Submitted prompt: ${recovery.content}`;
      const actions = document.createElement("div");
      actions.className = "d-flex flex-wrap gap-2";
      const refresh = document.createElement("button");
      refresh.type = "button";
      refresh.className = "btn btn-sm btn-outline-secondary";
      refresh.textContent = "Refresh history";
      refresh.hidden = recovery.failed;
      refresh.disabled = !selectedConversationId || lifecycleLoading;
      refresh.addEventListener("click", async () => {
        if (!selectedConversationId) return;
        const loaded = await selectConversation(selectedConversationId, { discardDraft: true, force: true, preserveDraft: true });
        if (loaded) {
          recoveryNeedsRefresh = false;
          renderRecovery();
          updateLifecycleControls();
        }
      });
      const reuse = document.createElement("button");
      reuse.type = "button";
      reuse.className = "btn btn-sm btn-outline-danger";
      reuse.textContent = "Use this prompt again";
      reuse.disabled = recoveryNeedsRefresh || (!recovery.failed && selectedConversationId !== recovery.conversationId);
      reuse.addEventListener("click", () => {
        if (!recovery.failed && selectedConversationId !== recovery.conversationId) return;
        const warning = recovery.failed
          ? "Restore this failed prompt to the composer?"
          : "The previous response may still be saving. Resending may duplicate a completed request. Continue?";
        if (!window.confirm(warning)) return;
        if (prompt.value && !window.confirm("Replace the current draft with the submitted prompt?")) return;
        prompt.value = recovery.content;
        promptRevision += 1;
        autosizePrompt();
        prompt.focus();
      });
      actions.append(refresh, reuse);
      recoveryPanel.append(message, submitted, actions);
    }

    function addContextResetNotice() {
      const notice = document.createElement("p");
      notice.className = "context-reset-notice";
      notice.textContent = "Saved messages are shown here. Earlier messages are not included in a new session’s AI context.";
      conversation.appendChild(notice);
      refreshEmptyState();
    }

    async function recoverHistory() {
      if (!selectedConversationId) return false;
      recoveryNeedsRefresh = true;
      updateLifecycleControls();
      const loaded = await selectConversation(selectedConversationId, { discardDraft: true, force: true, preserveDraft: true });
      if (loaded) {
        recoveryNeedsRefresh = false;
        renderRecovery();
        updateLifecycleControls();
        return true;
      }
      updateLifecycleControls();
      return false;
    }

    function handleSocketMessage(ws, event) {
      let payload;
      try { payload = JSON.parse(event.data); }
      catch {
        setSocketBanner("danger", "Received an unreadable chat event. Reconnect to continue.", true);
        return;
      }
      if (!payload || typeof payload !== "object" || typeof payload.type !== "string") return;
      const turnId = payload.turn_id;
      if (payload.type === "conversation_created") {
        if (activeTurn?.legacyCreation && payload.conversation?.id) {
          activeTurn.conversationId = payload.conversation.id;
          selectedConversationId = payload.conversation.id;
          refreshConversations().catch((error) => setSocketBanner("warning", error.message, true));
        }
        return;
      }
      if (payload.type === "error" && (turnId === "invalid" || turnId === "unavailable")) {
        setSocketBanner("danger", payload.content || "Chat is unavailable.", true);
        if (activeTurn && !activeTurn.turnId) {
          recovery = { content: activeTurn.content, conversationId: activeTurn.conversationId, failed: true };
          recoveryNeedsRefresh = false;
          renderRecovery();
          activeTurn = null;
          setChatBusy(false);
        }
        if (turnId === "invalid") {
          historyReady = false;
          retryHistory();
        }
        updateLifecycleControls();
        return;
      }
      if (payload.type === "turn_started") {
        if (!activeTurn || activeTurn.turnId || activeTurn.conversationId !== selectedConversationId || payload.conversation_id !== selectedConversationId || !turnId) return;
        activeTurn.turnId = turnId;
        const turn = createTurn(turnId);
        turn.status.textContent = "Starting response…";
        return;
      }
      if (!activeTurn || !turnId || activeTurn.turnId !== turnId || activeTurn.conversationId !== selectedConversationId || ws !== socket) return;
      if (payload.type === "status") {
        if (payload.phase === "model") addTimelineItem(turnId, payload.content || "Thinking…");
      } else if (payload.type === "tool_call_started") {
        updateTool(turnId, payload, true, activeTurn.toolPosition++);
      } else if (payload.type === "tool_call_completed") {
        updateTool(turnId, payload, false, Math.max(0, activeTurn.toolPosition - 1));
      } else if (payload.type === "usage" && payload.scope === "total") {
        updateUsage(turnId, payload);
      } else if (payload.type === "assistant_message") {
        setAssistantMessage(turnId, payload.content);
      } else if (payload.type === "sources") {
        renderSources(turnId, Array.isArray(payload.items) ? payload.items : []);
      } else if (payload.type === "error") {
        setTurnError(turnId, payload.content || "Chat failed.");
      } else if (payload.type === "turn_completed") {
        if (payload.success !== true) {
          if (turns.get(turnId)?.answer.textContent === "") setTurnError(turnId, "The response failed before it could be saved.");
          interruptRunningTools(turnId);
          if (turns.has(turnId)) turns.get(turnId).status.textContent = "Response failed. The prompt was not saved.";
          recovery = { content: activeTurn.content, conversationId: selectedConversationId, failed: true };
          recoveryNeedsRefresh = false;
          renderRecovery();
        } else if (turns.has(turnId)) {
          turns.get(turnId).status.textContent = "Answer ready.";
          chatAnnouncements.textContent = "Answer ready.";
          refreshConversations().catch((error) => setSocketBanner("warning", error.message, true));
        }
        activeTurn = null;
        setChatBusy(false);
        updateLifecycleControls();
      }
    }

    function preserveUncertainTurn() {
      if (!activeTurn) return;
      recovery = { content: activeTurn.content, conversationId: activeTurn.conversationId };
      recoveryNeedsRefresh = true;
      if (activeTurn.turnId) {
        interruptRunningTools(activeTurn.turnId);
        const turn = turns.get(activeTurn.turnId);
        if (turn) turn.status.textContent = "Connection lost. The previous response may still be saving.";
      }
      activeTurn = null;
      setChatBusy(false);
      renderRecovery();
    }

    function createSocket() {
      const generation = ++socketGeneration;
      const ws = new WebSocket(`${protocol}://${window.location.host}${wsPath}`);
      socket = ws;
      ws.addEventListener("open", async () => {
        if (socket !== ws || generation !== socketGeneration) return;
        clearSocketBanner();
        connectionStatus.textContent = "Connected";
        if (generation > 1) {
          setSocketBanner("warning", "Connection restored. Reloading saved messages; earlier messages are not included in a new session’s AI context.");
          await recoverHistory();
          if (recovery && !recovery.failed) setSocketBanner("warning", "Connection restored. The previous response may still be saving. Refresh history before resending.", true);
          else clearSocketBanner();
        }
        updateLifecycleControls();
      });
      ws.addEventListener("close", () => {
        if (socket !== ws || generation !== socketGeneration) return;
        preserveUncertainTurn();
        setSocketBanner("warning", "Connection closed. Reconnect to continue.", true);
        updateLifecycleControls();
      });
      ws.addEventListener("error", () => {
        if (socket === ws && generation === socketGeneration) setSocketBanner("danger", "The chat connection encountered an error. Reconnect to continue.", true);
      });
      ws.addEventListener("message", (event) => {
        if (socket === ws && generation === socketGeneration) handleSocketMessage(ws, event);
      });
    }

    function reconnectSocket() {
      const previousSocket = socket;
      socket = null;
      preserveUncertainTurn();
      if (previousSocket && previousSocket.readyState !== WebSocket.CLOSED) previousSocket.close();
      createSocket();
      connectionStatus.textContent = "Connecting…";
      setSocketBanner("warning", "Connecting to chat…");
      updateLifecycleControls();
    }

    createSocket();

    newConversationButton.addEventListener("click", () => createConversation());
    renameConversationButton.addEventListener("click", () => {
      const item = conversations.find((conversation) => conversation.id === selectedConversationId);
      if (!item || lifecycleBusy()) return;
      renameInput.value = item.title;
      renameError.textContent = "";
      renameDialog.showModal();
      renameInput.focus();
      renameInput.select();
    });
    document.getElementById("rename-cancel").addEventListener("click", () => renameDialog.close());
    renameDialog.addEventListener("close", () => renameConversationButton.focus());
    renameForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      const title = renameInput.value.trim();
      if (!title || title.length > 120 || !selectedConversationId) {
        renameError.textContent = "Enter a name between 1 and 120 characters.";
        return;
      }
      const targetId = selectedConversationId;
      setLifecycleMutation(true);
      renameSaveButton.disabled = true;
      renameError.textContent = "";
      try {
        const updated = await api(`/conversations/${targetId}`, { method: "PATCH", body: JSON.stringify({ title }) });
        if (selectedConversationId === targetId) {
          conversations = conversations.map((item) => item.id === targetId ? updated : item);
          renderConversationList();
        }
        renameDialog.close();
        await refreshConversations().catch((error) => setHistoryError(`Conversation renamed, but history could not refresh: ${error.message}`, () => retryHistory()));
      } catch (error) {
        if (error.status === 404) {
          selectedConversationId = null;
          historyReady = false;
          clearConversationView();
          renameDialog.close();
          setHistoryError("This conversation is unavailable. Refresh history and choose another conversation.", () => retryHistory());
          await refreshConversations().catch(() => {});
        } else {
          renameError.textContent = error.message || "Unable to rename this conversation.";
        }
      } finally {
        renameSaveButton.disabled = false;
        setLifecycleMutation(false);
      }
    });
    deleteConversationButton.addEventListener("click", async () => {
      const target = conversations.find((item) => item.id === selectedConversationId);
      if (!target || lifecycleBusy()) return;
      if (prompt.value.trim() && !window.confirm("Discard the unsent draft and delete this conversation?")) return;
      if (!window.confirm(`Delete “${target.title}”? This deletes only chat history, never Paperless documents.`)) return;
      const targetId = selectedConversationId;
      setLifecycleMutation(true);
      clearHistoryError();
      try {
        await api(`/conversations/${targetId}`, { method: "DELETE" });
        selectedConversationId = null;
        historyReady = false;
        clearConversationView();
        prompt.value = "";
        conversations = conversations.filter((item) => item.id !== targetId);
        renderConversationList();
        setLifecycleMutation(false);
        let historyRefreshed = true;
        await refreshConversations().catch((error) => {
          historyRefreshed = false;
          setHistoryError(`Conversation deleted, but history could not refresh: ${error.message}`, () => retryHistory());
        });
        if (!historyRefreshed) return;
        if (conversations.length) await selectConversation(conversations[0].id, { discardDraft: true });
        else await createConversation();
      } catch (error) {
        if (error.status === 404) {
          selectedConversationId = null;
          historyReady = false;
          clearConversationView();
          setHistoryError("This conversation is unavailable. Refresh history and choose another conversation.", () => retryHistory());
          await refreshConversations().catch(() => {});
        } else {
          setHistoryError(`Unable to delete this conversation: ${error.message}`, () => retryHistory());
        }
      } finally {
        setLifecycleMutation(false);
      }
    });
    historyRefreshButton.addEventListener("click", async () => {
      clearHistoryError();
      await Promise.all([
        refreshConversations().catch((error) => setHistoryError(`Unable to refresh history: ${error.message}`, () => retryHistory())),
        loadArchiveStatus(),
      ]);
    });
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      if (composing) return;
      const content = prompt.value.trim();
      if (!content) return;
      if (!historyReady || lifecycleLoading || chatBusy || recoveryNeedsRefresh || !selectedConversationId) {
        historyFeedback.textContent = "Conversation history is still loading or the previous response needs a history refresh.";
        return;
      }
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        setSocketBanner("warning", "The connection is not open. Reconnect to continue.", true);
        return;
      }
      const conversationId = selectedConversationId;
      try {
        socket.send(JSON.stringify({ content, conversation_id: conversationId }));
      } catch {
        setSocketBanner("danger", "The message could not be sent. Your draft was kept; reconnect and try again.", true);
        return;
      }
      activeTurn = { content, conversationId, turnId: null, toolPosition: 0 };
      recovery = null;
      recoveryNeedsRefresh = false;
      renderRecovery();
      setChatBusy(true);
      addUserBubble(content);
      prompt.value = "";
      autosizePrompt();
      prompt.focus();
      updateLifecycleControls();
    });

    prompt.addEventListener("compositionstart", () => { composing = true; });
    prompt.addEventListener("compositionend", () => { composing = false; });
    prompt.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || composing || event.isComposing || event.keyCode === 229) return;
      if (event.altKey && !event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      form.requestSubmit();
    });

    prompt.addEventListener("input", updateLifecycleControls);

    loadArchiveStatus();
    refreshConversations()
      .then(() => {
        historyReady = true;
        renderConversationList();
        updateLifecycleControls();
        return conversations.length ? selectConversation(conversations[0].id, { discardDraft: true, preserveDraft: true }) : createConversation();
      })
      .catch((error) => setHistoryError(`Unable to load chat history: ${error.message}`, () => retryHistory()));
  </script>
</body>
</html>
        """
    )


@app.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket) -> None:
    """Interactive Paperless copilot over WebSocket."""
    await websocket.accept()
    try:
        chat_copilot = _get_chat_copilot()
    except RuntimeError:
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
    active_conversation_id: str | None = None
    owner_id = _chat_owner(websocket.headers)
    chat_store = _require_chat_store()
    try:
        while True:
            incoming = await websocket.receive_text()
            try:
                payload = json.loads(incoming)
            except json.JSONDecodeError:
                payload = {"content": incoming}
            if not isinstance(payload, dict):
                await websocket.send_json(
                    {
                        "type": "error",
                        "turn_id": "invalid",
                        "content": "Invalid chat message.",
                    }
                )
                continue
            user_message = str(payload.get("content") or "").strip()
            if not user_message:
                continue
            conversation_id = payload.get("conversation_id") or active_conversation_id
            if conversation_id is None:
                conversation = await chat_store.create_conversation(owner_id)
                conversation_id = conversation["id"]
                await websocket.send_json(
                    {"type": "conversation_created", "conversation": conversation}
                )
            conversation_id = str(conversation_id)
            try:
                conversation_id = _chat_conversation_id(conversation_id)
            except HTTPException:
                await websocket.send_json(
                    {
                        "type": "error",
                        "turn_id": "invalid",
                        "content": "Invalid conversation ID.",
                    }
                )
                continue
            if conversation_id != active_conversation_id:
                if (
                    await chat_store.load_conversation(conversation_id, owner_id)
                    is None
                ):
                    await websocket.send_json(
                        {
                            "type": "error",
                            "turn_id": "invalid",
                            "content": "Conversation not found.",
                        }
                    )
                    continue
                history = []
                active_conversation_id = conversation_id
            turn_id = uuid.uuid4().hex
            await websocket.send_json(
                {
                    "type": "turn_started",
                    "turn_id": turn_id,
                    "conversation_id": conversation_id,
                }
            )

            async def emit(event: dict) -> None:
                payload = {"turn_id": turn_id, **event}
                await websocket.send_json(payload)

            try:
                result = await chat_copilot.run_turn(
                    user_message, history, event_callback=emit
                )
            except Exception as exc:
                log.exception("Chat turn failed")
                await emit(
                    {
                        "type": "error",
                        "content": f"Chat request failed: {type(exc).__name__}: {exc}",
                    }
                )
                await emit({"type": "turn_completed", "success": False})
                continue
            try:
                sources = await _build_chat_sources(result.sources)
                saved = await chat_store.append_turn(
                    conversation_id,
                    owner_id,
                    user_message,
                    result.reply or "(no response)",
                    result.tool_activity,
                    chat_copilot._config.chat_model,
                    result.usage,
                    sources,
                )
            except Exception as exc:
                log.exception("Chat history persistence failed")
                await emit(
                    {
                        "type": "error",
                        "content": f"Chat history could not be saved: {type(exc).__name__}: {exc}",
                    }
                )
                await emit({"type": "turn_completed", "success": False})
                continue
            if not saved:
                await emit({"type": "error", "content": "Conversation was deleted."})
                await emit({"type": "turn_completed", "success": False})
                continue
            history = result.history
            await emit(
                {
                    "type": "assistant_message",
                    "content": result.reply or "(no response)",
                }
            )
            await emit(
                {
                    "type": "usage",
                    "scope": "total",
                    "model": chat_copilot._config.chat_model,
                    "available": bool(result.usage),
                    **(result.usage or {}),
                }
            )
            await emit({"type": "sources", "items": sources})
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


def _worker_health_snapshot() -> tuple[bool, dict]:
    if _config is None:
        return False, {"ready": False, "error": "config-unavailable", "stages": {}}
    if _worker_setup_error is not None:
        return False, {"ready": False, "error": _worker_setup_error, "stages": {}}
    if not _worker_ready:
        return False, {"ready": False, "error": "worker-not-ready", "stages": {}}

    now = time.time()
    stale_after_seconds = max(60, (_config.poll_interval * 2) + 30)
    stages: dict[str, dict[str, float | bool]] = {}
    healthy = True
    for stage, ts in _worker_heartbeats.items():
        age_seconds = max(0.0, now - ts)
        stage_ok = age_seconds <= stale_after_seconds
        healthy = healthy and stage_ok
        stages[stage] = {
            "healthy": stage_ok,
            "age_seconds": round(age_seconds, 1),
        }
    return healthy, {
        "ready": True,
        "stale_after_seconds": stale_after_seconds,
        "stages": stages,
    }


@app.get("/health")
async def health() -> JSONResponse:
    if _queues is not None:
        pending = await _queues.pending_count()
    else:
        pending = {"ocr": 0, "metadata": 0, "embed": 0, "refresh": 0}

    worker_ok, worker = _worker_health_snapshot()
    body = {
        "status": "ok" if worker_ok else "degraded",
        "pending": pending,
        "worker": worker,
    }
    return JSONResponse(status_code=200 if worker_ok else 503, content=body)
