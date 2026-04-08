"""Process-backed local search inference for the webhook listener."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import time
from multiprocessing.connection import Connection
from typing import Any, Callable

from paperless_ai.search.embedder_types import EmbeddingResult

log = logging.getLogger(__name__)

_Request = dict[str, Any]
_Response = dict[str, Any]
_WorkerTarget = Callable[[Connection, int], None]


def _payload_summary(payload: _Request) -> str:
    action = str(payload.get("action"))
    if action == "warmup":
        return (
            f"action={action} preload_embed={bool(payload.get('preload_embed', True))} "
            f"preload_rerank={bool(payload.get('preload_rerank', True))}"
        )
    if action == "embed_query":
        return f"action={action} query_len={len(str(payload.get('query', '')))}"
    if action == "rerank":
        passages = payload.get("passages") or []
        return (
            f"action={action} query_len={len(str(payload.get('query', '')))} "
            f"passages={len(passages)} normalize={bool(payload.get('normalize', False))}"
        )
    return f"action={action}"


def _configure_worker_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _worker_main(conn: Connection, idle_timeout_seconds: int) -> None:
    from paperless_ai.search.embedder import LocalLazySearchEmbedder

    _configure_worker_logging()
    log.info(
        "Local search worker booting pid=%s ppid=%s idle_timeout=%ss",
        os.getpid(),
        os.getppid(),
        idle_timeout_seconds,
    )
    embedder = LocalLazySearchEmbedder()
    while True:
        if not conn.poll(idle_timeout_seconds):
            log.info(
                "Local search worker idle for >%ss — exiting process pid=%s",
                idle_timeout_seconds,
                os.getpid(),
            )
            break

        request = conn.recv()
        action = request.get("action")
        log.info(
            "Local search worker pid=%s received %s",
            os.getpid(),
            _payload_summary(request),
        )
        if action == "shutdown":
            log.info("Local search worker pid=%s received shutdown", os.getpid())
            conn.send({"ok": True})
            break

        try:
            started = time.monotonic()
            if action == "warmup":
                if bool(request.get("preload_embed", True)):
                    embedder._get_model()
                if bool(request.get("preload_rerank", True)):
                    embedder._get_reranker(embedder.LOCAL_RERANKER_MODEL_NAME)
                conn.send({"ok": True})
                log.info(
                    "Local search worker pid=%s completed warmup in %.2fs",
                    os.getpid(),
                    time.monotonic() - started,
                )
                continue

            if action == "embed_query":
                result = asyncio.run(embedder.embed_query(str(request["query"])))
                conn.send(
                    {
                        "ok": True,
                        "dense": result.dense,
                        "sparse_indices": result.sparse_indices,
                        "sparse_values": result.sparse_values,
                    }
                )
                log.info(
                    "Local search worker pid=%s completed embed_query in %.2fs",
                    os.getpid(),
                    time.monotonic() - started,
                )
                continue

            if action == "rerank":
                scores = asyncio.run(
                    embedder.rerank(
                        str(request["query"]),
                        list(request["passages"]),
                        model_name=str(request["model_name"]),
                        normalize=bool(request.get("normalize", False)),
                    )
                )
                conn.send({"ok": True, "scores": scores})
                log.info(
                    "Local search worker pid=%s completed rerank in %.2fs",
                    os.getpid(),
                    time.monotonic() - started,
                )
                continue

            raise ValueError(f"Unknown local-search action: {action}")
        except Exception as exc:
            log.exception(
                "Local search worker pid=%s failed while handling %s",
                os.getpid(),
                _payload_summary(request),
            )
            conn.send({"ok": False, "error": repr(exc)})

    log.info("Local search worker pid=%s shutting down", os.getpid())
    conn.close()


class ProcessLocalSearchEmbedder:
    """Lazily starts a child process that owns local search models."""

    MODEL_NAME = "BAAI/bge-m3"
    LOCAL_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        *,
        idle_timeout_seconds: int = 300,
        start_method: str = "spawn",
        worker_target: _WorkerTarget | None = None,
    ) -> None:
        self._idle_timeout_seconds = idle_timeout_seconds
        self._ctx = mp.get_context(start_method)
        self._start_method = start_method
        self._worker_target = worker_target or _worker_main
        self._lock = asyncio.Lock()
        self._process: mp.Process | None = None
        self._conn: Connection | None = None

    def _is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def _start_worker(self) -> None:
        if self._is_running():
            return
        self._close_dead_worker()
        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=self._worker_target,
            args=(child_conn, self._idle_timeout_seconds),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process = process
        self._conn = parent_conn
        log.info(
            "Started local search worker process pid=%s parent_pid=%s start_method=%s idle_timeout=%ss",
            process.pid,
            os.getpid(),
            self._start_method,
            self._idle_timeout_seconds,
        )

    def _close_dead_worker(self) -> None:
        process = self._process
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._conn = None
        if process is not None:
            process.join(timeout=0)
            log.info(
                "Closed local search worker handle pid=%s alive=%s exitcode=%s",
                process.pid,
                process.is_alive(),
                process.exitcode,
            )
        self._process = None

    def _request(self, payload: _Request) -> _Response:
        self._start_worker()
        assert self._conn is not None
        process = self._process
        assert process is not None
        log.info(
            "Parent pid=%s sending to local search worker pid=%s %s",
            os.getpid(),
            process.pid,
            _payload_summary(payload),
        )
        started = time.monotonic()
        try:
            self._conn.send(payload)
            response = self._conn.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            log.warning(
                "Local search worker pid=%s communication failed with %s; restarting worker",
                process.pid,
                type(exc).__name__,
            )
            self._close_dead_worker()
            self._start_worker()
            assert self._conn is not None
            process = self._process
            assert process is not None
            log.info(
                "Parent pid=%s retrying against local search worker pid=%s %s",
                os.getpid(),
                process.pid,
                _payload_summary(payload),
            )
            self._conn.send(payload)
            response = self._conn.recv()
        if not response.get("ok"):
            log.error(
                "Local search worker pid=%s returned error for %s: %s",
                process.pid,
                _payload_summary(payload),
                response.get("error", "unknown error"),
            )
            raise RuntimeError(str(response.get("error", "local search worker failed")))
        log.info(
            "Parent pid=%s received response from local search worker pid=%s in %.2fs for %s",
            os.getpid(),
            process.pid,
            time.monotonic() - started,
            _payload_summary(payload),
        )
        if self._process is not None and not self._process.is_alive():
            log.info(
                "Local search worker pid=%s exited after responding; cleaning up handle",
                self._process.pid,
            )
            self._close_dead_worker()
        return response

    async def embed_query(self, query: str) -> EmbeddingResult:
        async with self._lock:
            response = await asyncio.to_thread(
                self._request,
                {"action": "embed_query", "query": query},
            )
        return EmbeddingResult(
            dense=list(response["dense"]),
            sparse_indices=list(response.get("sparse_indices", [])),
            sparse_values=list(response.get("sparse_values", [])),
        )

    async def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        model_name: str,
        normalize: bool = False,
    ) -> list[float]:
        async with self._lock:
            response = await asyncio.to_thread(
                self._request,
                {
                    "action": "rerank",
                    "query": query,
                    "passages": passages,
                    "model_name": model_name,
                    "normalize": normalize,
                },
            )
        return [float(score) for score in response["scores"]]

    async def warmup(
        self,
        *,
        preload_embed: bool = True,
        preload_rerank: bool = True,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._request,
                {
                    "action": "warmup",
                    "preload_embed": preload_embed,
                    "preload_rerank": preload_rerank,
                },
            )

    async def aclose(self) -> None:
        async with self._lock:
            if not self._is_running() or self._conn is None:
                self._close_dead_worker()
                return
            process = self._process
            assert process is not None
            log.info(
                "Parent pid=%s shutting down local search worker pid=%s",
                os.getpid(),
                process.pid,
            )
            try:
                await asyncio.to_thread(self._conn.send, {"action": "shutdown"})
                await asyncio.to_thread(self._conn.recv)
            except (EOFError, BrokenPipeError, OSError):
                pass
            finally:
                self._close_dead_worker()
                if process is not None:
                    process.join(timeout=1)
                    if process.is_alive():
                        log.warning(
                            "Local search worker pid=%s did not exit cleanly; terminating",
                            process.pid,
                        )
                        process.terminate()
                        process.join(timeout=5)
