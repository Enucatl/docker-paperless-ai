"""Interactive copilot and search app hosted by the always-on AI service."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import niquests
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from paperless_common.secrets import read_secret

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.runtime import initialize_paperless, workers
from paperless_common.paperless import PaperlessClient
from paperless_common.telemetry import setup_telemetry
from paperless_common.queue import TaskQueues

if TYPE_CHECKING:
    from paperless_ai.search.chat_agent import ChatCopilot
    from paperless_ai.search.chat_store import ChatStore

log = logging.getLogger(__name__)

_queues: TaskQueues | None = None
_paperless_client: PaperlessClient | None = None
_chat_copilot: ChatCopilot | None = None
_chat_store: ChatStore | None = None
_config: AgentConfig | None = None
_worker_tasks: list[asyncio.Task] = []
_worker_heartbeats: dict[str, float] = {"ocr": 0.0, "metadata": 0.0}
_worker_ready: bool = False
_worker_setup_error: str | None = None


def _is_unavailable_chat_source_error(exc: Exception) -> bool:
    """Return whether a document metadata failure should retain an unavailable source."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {401, 403, 404, 408, 429} or status_code >= 500
    return isinstance(exc, niquests.RequestException)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _queues, _paperless_client, _chat_copilot, _chat_store, _config
    global _worker_tasks, _worker_heartbeats, _worker_ready, _worker_setup_error

    redis_url = os.environ.get("REDIS_URL", "redis://broker:6379/1")
    paperless_url = os.environ.get("PAPERLESS_URL")
    paperless_token = read_secret("PAPERLESS_TOKEN")
    log.info(
        "Startup config: redis=%s paperless_url=%r paperless_token=%s",
        redis_url,
        paperless_url,
        "loaded" if paperless_token else "missing",
    )
    if not paperless_url:
        raise RuntimeError("PAPERLESS_URL is not set for the copilot service")
    if not paperless_token:
        raise RuntimeError(
            "PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set for the copilot service"
        )

    _paperless_client = PaperlessClient(paperless_url, paperless_token)
    log.info("Paperless keyword search enabled (%s)", paperless_url)

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

    from paperless_ai.core.runner import clear_shutdown_request

    _worker_ready = False
    _worker_setup_error = None
    _worker_heartbeats = {"ocr": 0.0, "metadata": 0.0}
    _worker_tasks = []
    clear_shutdown_request()

    try:
        log.info("Checking Paperless API connectivity and managed resources...")
        field_ids = await initialize_paperless(_paperless_client, _config)

        _chat_copilot = None
        log.info("Chat copilot enabled lazily")

        def _mark_worker_heartbeat(stage: str) -> None:
            _worker_heartbeats[stage] = time.time()

        async with workers(
            _paperless_client, _config, _queues, field_ids, _mark_worker_heartbeat
        ) as tasks:
            _worker_tasks = tasks
            _worker_ready = True
            log.info("Copilot service ready")
            yield

    finally:
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
app.mount(
    "/assets", StaticFiles(directory=Path(__file__).with_name("assets")), name="assets"
)


def _get_chat_copilot() -> ChatCopilot:
    global _chat_copilot
    if _config is None or _paperless_client is None:
        raise RuntimeError("Paperless chat is not configured")
    if _chat_copilot is None:
        from paperless_ai.search.chat_agent import ChatCopilot

        _chat_copilot = ChatCopilot(_config, _paperless_client)
        log.info("Chat copilot initialized")
    return _chat_copilot


@app.get("/metadata/available")
async def available_metadata() -> JSONResponse:
    """Return exact metadata names available for agentic search pre-filtering."""
    if _paperless_client is None:
        raise HTTPException(status_code=503, detail="Paperless client not configured")
    client = _paperless_client.metadata_snapshot()
    return JSONResponse(content=await client.get_available_metadata())


def _document_detail_url(doc_id: int) -> str:
    return f"/documents/{doc_id}/detail"


def _document_thumb_url(doc_id: int) -> str:
    return f"/api/documents/{doc_id}/thumb/"


def _document_preview_url(doc_id: int) -> str:
    return f"/api/documents/{doc_id}/preview/"


async def _build_chat_sources(source_flags: dict[int, dict[str, bool]]) -> list[dict]:
    """Build sorted live cards through the same refresh path as saved cards."""
    return await _restore_chat_sources(
        [
            {
                "id": doc_id,
                "title": f"Document {doc_id}",
                "matched": bool(flags.get("matched")),
                "inspected": bool(flags.get("inspected")),
            }
            for doc_id, flags in sorted(
                source_flags.items(),
                key=lambda item: (not item[1].get("inspected", False), item[0]),
            )
        ]
    )


async def _restore_chat_sources(sources: list[dict]) -> list[dict]:
    """Refresh cards with one request-local lookup per distinct document."""
    client = _paperless_client.metadata_snapshot() if _paperless_client else None
    documents: dict[int, dict | None] = {}
    items: list[dict] = []
    for source in sources:
        doc_id = source["id"]
        if doc_id not in documents:
            try:
                documents[doc_id] = (
                    await client.get_document_chat_metadata(doc_id)
                    if client is not None
                    else None
                )
            except Exception as exc:
                if not _is_unavailable_chat_source_error(exc):
                    raise
                log.info("Could not refresh chat source %s: %s", doc_id, exc)
                documents[doc_id] = None
        metadata = documents[doc_id]
        items.append(
            {
                **source,
                **(metadata or {}),
                "available": metadata is not None,
                "detail_url": _document_detail_url(doc_id),
                "thumb_url": _document_thumb_url(doc_id),
                "preview_url": _document_preview_url(doc_id),
            }
        )
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
    sources = iter(
        await _restore_chat_sources(
            [
                source
                for message in conversation["messages"]
                for source in message["sources"]
            ]
        )
    )
    for message in conversation["messages"]:
        message["sources"] = [next(sources) for _ in message["sources"]]
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
async def chat_ui() -> FileResponse:
    """Serve the packaged chat browser UI."""
    return FileResponse(Path(__file__).with_name("assets") / "chat.html")


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
        pending = {"ocr": 0, "metadata": 0}

    worker_ok, worker = _worker_health_snapshot()
    body = {
        "status": "ok" if worker_ok else "degraded",
        "pending": pending,
        "worker": worker,
    }
    return JSONResponse(status_code=200 if worker_ok else 503, content=body)
