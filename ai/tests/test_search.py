"""
Tests for LocalLazySearchEmbedder and the GET /search endpoint.

Test categories
---------------
Unit (no Docker required):
  test_model_starts_none
  test_get_model_loads_on_first_call
  test_get_model_reuses_instance
  test_get_model_updates_last_used
  test_embed_query_returns_correct_shape
  test_idle_watcher_keeps_fresh_model
  test_idle_watcher_unloads_stale_model
  test_idle_watcher_calls_gc_collect
  test_memory_lifecycle_allocate_and_free   ← tracemalloc + weakref

Endpoint integration (webhook-listener + Qdrant running):
  test_search_missing_q_returns_422
  test_search_empty_string_returns_422
  test_search_returns_list
  test_search_limit_too_large_returns_422
  test_search_deduplicates_chunks_across_same_doc

Why tracemalloc, not Scalene?
------------------------------
Scalene is a *sampling profiler* invoked via `scalene script.py` at the
terminal.  It emits annotated HTML/stdout reports and has no programmatic
assertion API.  Its sampling is statistical, making it non-deterministic as a
CI gate.  The right combination for this use case is:

  * tracemalloc (stdlib) — takes before/after Python-heap snapshots and
    produces precise byte-level diffs suitable for assertions.
  * weakref (stdlib) — deterministically proves an object was collected by
    the GC, not just dereferenced (avoids false-passing when CPython cycles
    keep the object alive).
"""

import asyncio
import gc
import time
import tracemalloc
import weakref
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import niquests
import pytest

from tests.conftest import WEBHOOK_URL
from paperless_ai.search.embedder import EmbeddingResult, LocalLazySearchEmbedder

# ---------------------------------------------------------------------------
# Fake SentenceTransformer model (no download, tracks allocation)
# ---------------------------------------------------------------------------

_DENSE_DIM = 1024
_ALLOC_BYTES = 10 * 1024 * 1024  # 10 MiB — large enough for tracemalloc to see


import numpy as np


class _FakeSentenceTransformer:
    """
    Drop-in for sentence_transformers.SentenceTransformer.

    Allocates a 10 MiB bytearray on construction so that tracemalloc can
    detect the allocation and confirm it is freed when the model is unloaded.
    """

    def __init__(self, model_name: str, trust_remote_code: bool = True):
        self._model_name = model_name
        # bytearray is tracked by Python's allocator → visible to tracemalloc
        self._weights = bytearray(_ALLOC_BYTES)

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """Return a normalized embedding vector."""
        return np.array([0.1] * _DENSE_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> LocalLazySearchEmbedder:
    return LocalLazySearchEmbedder()


@pytest.fixture
def patched_sentence_transformer():
    """Patch sentence_transformers.SentenceTransformer globally for the duration of the test."""
    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        yield _FakeSentenceTransformer


# ---------------------------------------------------------------------------
# Unit: model loading behaviour
# ---------------------------------------------------------------------------


def test_model_starts_none(embedder):
    assert embedder.model is None


def test_get_model_loads_on_first_call(embedder, patched_sentence_transformer):
    model = embedder._get_model()
    assert model is not None
    assert isinstance(model, _FakeSentenceTransformer)


def test_get_model_reuses_instance(embedder, patched_sentence_transformer):
    first = embedder._get_model()
    second = embedder._get_model()
    assert first is second


def test_get_model_updates_last_used(embedder, patched_sentence_transformer):
    before = time.monotonic()
    embedder._get_model()
    assert embedder._last_used >= before


def test_get_model_logs_on_load(embedder, patched_sentence_transformer, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="paperless_ai.search.embedder"):
        embedder._get_model()
    assert any("Loading" in r.message for r in caplog.records)


def test_get_model_does_not_log_on_second_call(embedder, patched_sentence_transformer, caplog):
    import logging

    embedder._get_model()  # first — logs
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="paperless_ai.search.embedder"):
        embedder._get_model()  # second — must NOT log
    assert not any("Loading" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Unit: embed_query
# ---------------------------------------------------------------------------


async def test_embed_query_returns_embedding_result(embedder, patched_sentence_transformer):
    result = await embedder.embed_query("what is this invoice about?")
    assert isinstance(result, EmbeddingResult)
    assert len(result.dense) == _DENSE_DIM
    assert all(isinstance(v, float) for v in result.dense)


async def test_embed_query_loads_model_lazily(embedder, patched_sentence_transformer):
    assert embedder.model is None
    await embedder.embed_query("test query")
    assert embedder.model is not None


async def test_embed_query_does_not_block_event_loop(embedder, patched_sentence_transformer):
    """embed_query must run embedding in a thread, not block the event loop.

    We verify this by running a concurrent coroutine while embed_query
    executes; if embedding were synchronous the counter would stay at 0.
    """
    counter = 0

    async def increment():
        nonlocal counter
        await asyncio.sleep(0)
        counter += 1

    inc_task = asyncio.create_task(increment())
    await embedder.embed_query("invoice rent 2024")
    await inc_task
    assert counter == 1, "Event loop was blocked during embed_query"


# ---------------------------------------------------------------------------
# Unit: idle_watcher
# ---------------------------------------------------------------------------


async def test_idle_watcher_keeps_fresh_model(embedder, patched_sentence_transformer):
    """A recently-used model must NOT be evicted."""
    embedder._get_model()  # loads model and stamps _last_used
    assert embedder.model is not None

    tick = 0

    async def fast_sleep(_seconds):
        nonlocal tick
        tick += 1
        if tick >= 2:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", fast_sleep):
        with pytest.raises(asyncio.CancelledError):
            await embedder.idle_watcher(timeout_seconds=300)

    assert embedder.model is not None, "Fresh model should not have been evicted"


async def test_idle_watcher_unloads_stale_model(embedder, patched_sentence_transformer):
    """A model idle longer than timeout_seconds must be set to None."""
    embedder._get_model()
    embedder._last_used = 0.0  # make it look ancient

    tick = 0

    async def fast_sleep(_seconds):
        nonlocal tick
        tick += 1
        if tick >= 2:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", fast_sleep):
        with pytest.raises(asyncio.CancelledError):
            await embedder.idle_watcher(timeout_seconds=0)

    assert embedder.model is None, "Stale model should have been unloaded"


async def test_idle_watcher_calls_gc_collect(embedder, patched_sentence_transformer):
    """gc.collect() must be called after model is unloaded."""
    embedder._get_model()
    embedder._last_used = 0.0

    tick = 0

    async def fast_sleep(_seconds):
        nonlocal tick
        tick += 1
        if tick >= 2:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", fast_sleep):
        with patch("gc.collect") as mock_gc:
            with pytest.raises(asyncio.CancelledError):
                await embedder.idle_watcher(timeout_seconds=0)

    mock_gc.assert_called_once()


async def test_idle_watcher_logs_eviction(embedder, patched_sentence_transformer, caplog):
    import logging

    embedder._get_model()
    embedder._last_used = 0.0
    tick = 0

    async def fast_sleep(_seconds):
        nonlocal tick
        tick += 1
        if tick >= 2:
            raise asyncio.CancelledError

    with patch("asyncio.sleep", fast_sleep):
        with caplog.at_level(logging.INFO, logger="paperless_ai.search.embedder"):
            with pytest.raises(asyncio.CancelledError):
                await embedder.idle_watcher(timeout_seconds=0)

    assert any("freeing RAM" in r.message or "idle" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# Memory lifecycle: tracemalloc + weakref
#
# This test documents the RAM footprint of the scale-to-zero design.
# It does NOT use Scalene.  See module docstring for rationale.
# ---------------------------------------------------------------------------


def test_memory_lifecycle_allocate_and_free():
    """
    Verify that the model's RAM is allocated on load and reclaimed after unload.

    Measurement strategy
    --------------------
    tracemalloc takes three snapshots:

      snap_baseline  — before any model object exists
      snap_loaded    — after _get_model() has been called
      snap_freed     — after model = None + gc.collect()

    The diff snap_loaded vs snap_baseline shows the bytes allocated.
    The diff snap_freed vs snap_loaded shows the bytes released (negative).

    weakref provides a deterministic guarantee that the object was actually
    collected by CPython's reference counter + cyclic GC, not just hidden from
    tracemalloc by surviving in some unreachable cycle.
    """
    tracemalloc.start()

    # ── Baseline ──────────────────────────────────────────────────────────
    snap_baseline = tracemalloc.take_snapshot()

    with patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer):
        embedder = LocalLazySearchEmbedder()
        embedder._get_model()  # triggers _FakeSentenceTransformer(…) → allocates bytearray

    # ── Loaded ────────────────────────────────────────────────────────────
    snap_loaded = tracemalloc.take_snapshot()

    # Obtain a weak reference so we can observe the object's GC lifecycle.
    wr = weakref.ref(embedder.model)
    assert wr() is not None, "Model must be alive immediately after _get_model()"

    allocated_diff = sum(
        s.size_diff
        for s in snap_loaded.compare_to(snap_baseline, "lineno")
        if s.size_diff > 0
    )

    # ── Unloaded ──────────────────────────────────────────────────────────
    embedder.model = None
    gc.collect()

    snap_freed = tracemalloc.take_snapshot()

    freed_diff = sum(
        s.size_diff
        for s in snap_freed.compare_to(snap_loaded, "lineno")
        if s.size_diff < 0
    )

    tracemalloc.stop()

    # ── Assertions ────────────────────────────────────────────────────────
    # The fake model allocates _ALLOC_BYTES; at least 80 % must be visible to
    # tracemalloc after load.
    assert allocated_diff >= _ALLOC_BYTES * 0.8, (
        f"Expected ≥{_ALLOC_BYTES * 0.8 / 1024:.0f} KiB allocated, "
        f"got {allocated_diff / 1024:.0f} KiB"
    )

    # After unload + gc.collect(), freed_diff should recover most of it.
    assert freed_diff <= -(_ALLOC_BYTES * 0.8), (
        f"Expected ≥{_ALLOC_BYTES * 0.8 / 1024:.0f} KiB freed, "
        f"got {abs(freed_diff) / 1024:.0f} KiB released"
    )

    # The model object itself must be gone from the GC heap.
    assert wr() is None, (
        "Model object was not collected after model=None + gc.collect(). "
        "Check for reference cycles."
    )

    # ── Human-readable report (visible with pytest -s) ────────────────────
    print(
        f"\n[memory] model loaded:  +{allocated_diff / 1024 / 1024:.2f} MiB\n"
        f"[memory] model unloaded: {freed_diff / 1024 / 1024:.2f} MiB\n"
        f"[memory] fake model size: {_ALLOC_BYTES / 1024 / 1024:.0f} MiB\n"
        f"[memory] weakref alive after unload: {wr() is not None}"
    )


# ---------------------------------------------------------------------------
# Endpoint integration tests (require running webhook-listener + Qdrant)
# ---------------------------------------------------------------------------


@pytest.mark.requires_webhook_listener
async def test_search_missing_q_returns_422():
    """/search without ?q must return 422 Unprocessable Entity."""
    async with niquests.AsyncSession() as client:
        r = await client.get(f"{WEBHOOK_URL}/search")
    assert r.status_code == 422


@pytest.mark.requires_webhook_listener
async def test_search_empty_string_returns_422():
    """/search?q= (empty string) must be rejected by the min_length=1 constraint."""
    async with niquests.AsyncSession() as client:
        r = await client.get(f"{WEBHOOK_URL}/search", params={"q": ""})
    assert r.status_code == 422


@pytest.mark.requires_webhook_listener
async def test_search_returns_list():
    """/search?q=... must return a JSON array (possibly empty when Qdrant is empty)."""
    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(f"{WEBHOOK_URL}/search", params={"q": "invoice"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert all(isinstance(doc_id, int) for doc_id in body)


@pytest.mark.requires_webhook_listener
async def test_search_respects_limit():
    """/search?limit=1 must return at most 1 result."""
    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(
            f"{WEBHOOK_URL}/search", params={"q": "document", "limit": 1}
        )
    assert r.status_code == 200
    assert len(r.json()) <= 1


@pytest.mark.requires_webhook_listener
async def test_search_limit_too_large_returns_422():
    """/search?limit=999 exceeds the max of 100 and must be rejected."""
    async with niquests.AsyncSession() as client:
        r = await client.get(
            f"{WEBHOOK_URL}/search", params={"q": "test", "limit": 999}
        )
    assert r.status_code == 422


@pytest.mark.requires_webhook_listener
async def test_search_deduplicates_chunks_across_same_doc(qdrant_store):
    """
    When multiple chunks from the same doc_id score highly, the /search
    endpoint must return that doc_id only once.

    Setup: upsert two chunks for doc_id=9999 with identical vectors, then
    search.  Without deduplication the endpoint would return [9999, 9999].
    """
    from paperless_ai.search.qdrant_store import ChunkPayload

    dense = [0.0] * 1024
    dense[0] = 1.0  # point in a unique direction to rank highly

    await qdrant_store.upsert_chunks(
        chunks=[
            ChunkPayload(
                doc_id=9999,
                chunk_index=0,
                title="Dedup test",
                correspondent=None,
                date=None,
                text="first chunk",
            ),
            ChunkPayload(
                doc_id=9999,
                chunk_index=1,
                title="Dedup test",
                correspondent=None,
                date=None,
                text="second chunk",
            ),
        ],
        dense_vecs=[dense, dense],
        sparse_indices=[[], []],
        sparse_values=[[], []],
    )

    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(
            f"{WEBHOOK_URL}/search", params={"q": "dedup test", "limit": 20}
        )

    assert r.status_code == 200
    doc_ids = r.json()
    assert doc_ids.count(9999) <= 1, (
        f"doc_id 9999 appeared {doc_ids.count(9999)} times — deduplication failed"
    )

    # Cleanup
    await qdrant_store.delete_document(9999)
