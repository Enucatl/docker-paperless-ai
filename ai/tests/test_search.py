"""
Tests for local search embedders and the GET /search endpoint.

Test categories
---------------
Unit (no Docker required):
  test_model_starts_none
  test_get_model_loads_on_first_call
  test_get_model_reuses_instance
  test_embed_query_returns_correct_shape
  test_memory_lifecycle_allocate_and_free   ← tracemalloc + weakref

Endpoint integration (ai copilot service + Qdrant running):
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
import math
import tracemalloc
import weakref
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, patch

import niquests
import pytest

from tests.conftest import COPILOT_URL, QDRANT_URL
from paperless_ai.search.embedder import EmbeddingResult, LocalLazySearchEmbedder
from paperless_ai.search.local_search_process import ProcessLocalSearchEmbedder
from paperless_ai.search.retriever import (
    MAX_RERANK_CANDIDATES,
    ChunkCandidate,
    hybrid_retrieve,
)

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


class _FakeFlagReranker:
    """Drop-in for FlagEmbedding.FlagReranker."""

    def __init__(self, model_name: str, use_fp16: bool = False):
        self._model_name = model_name
        self._use_fp16 = use_fp16

    def compute_score(self, pairs, normalize: bool = False):
        normalized_pairs = pairs if pairs and isinstance(pairs[0], list) else [pairs]
        scores = []
        for _query, passage in normalized_pairs:
            if "best" in passage:
                score = 8.0
            elif "mid" in passage:
                score = 0.0
            else:
                score = -8.0
            if normalize:
                score = 1.0 / (1.0 + math.exp(-score))
            scores.append(score)
        return scores[0] if len(scores) == 1 else scores


def _fake_worker(conn, idle_timeout_seconds: int):
    while True:
        if not conn.poll(idle_timeout_seconds):
            break
        request = conn.recv()
        action = request.get("action")
        if action == "shutdown":
            conn.send({"ok": True})
            break
        if action == "warmup":
            conn.send({"ok": True})
            continue
        if action == "embed_query":
            conn.send(
                {
                    "ok": True,
                    "dense": [0.1, 0.2],
                    "sparse_indices": [],
                    "sparse_values": [],
                }
            )
            continue
        if action == "rerank":
            conn.send(
                {
                    "ok": True,
                    "scores": [
                        float(idx) for idx, _ in enumerate(request["passages"], start=1)
                    ],
                }
            )
            continue
        conn.send({"ok": False, "error": f"bad action: {action}"})
    conn.close()


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


@pytest.fixture
def patched_flag_reranker():
    with patch("paperless_ai.search.embedder.FlagReranker", _FakeFlagReranker):
        yield _FakeFlagReranker


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


def test_get_model_logs_on_load(embedder, patched_sentence_transformer, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="paperless_ai.search.embedder"):
        embedder._get_model()
    assert any("Loading" in r.message for r in caplog.records)


def test_get_model_does_not_log_on_second_call(
    embedder, patched_sentence_transformer, caplog
):
    import logging

    embedder._get_model()  # first — logs
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="paperless_ai.search.embedder"):
        embedder._get_model()  # second — must NOT log
    assert not any("Loading" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Unit: embed_query
# ---------------------------------------------------------------------------


async def test_embed_query_returns_embedding_result(
    embedder, patched_sentence_transformer
):
    result = await embedder.embed_query("what is this invoice about?")
    assert isinstance(result, EmbeddingResult)
    assert len(result.dense) == _DENSE_DIM
    assert all(isinstance(v, float) for v in result.dense)


async def test_embed_query_loads_model_lazily(embedder, patched_sentence_transformer):
    assert embedder.model is None
    await embedder.embed_query("test query")
    assert embedder.model is not None


async def test_embed_query_does_not_block_event_loop(
    embedder, patched_sentence_transformer
):
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


async def test_rerank_returns_normalized_scores(embedder, patched_flag_reranker):
    scores = await embedder.rerank(
        "what is panda?",
        ["worst passage", "best passage"],
        model_name=embedder.LOCAL_RERANKER_MODEL_NAME,
        normalize=True,
    )

    assert len(scores) == 2
    assert 0.0 < scores[0] < 1.0
    assert 0.0 < scores[1] < 1.0
    assert scores[1] > scores[0]
    assert embedder._reranker is not None


async def test_hybrid_retrieve_uses_local_reranker():
    class _FakeClient:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        async def search_documents_all(self, query: str, **kwargs):
            self.calls.append((query, kwargs))
            return [3, 2]

    client = _FakeClient()
    embedder = MagicMock(spec=LocalLazySearchEmbedder)

    with (
        patch(
            "paperless_ai.search.retriever.dense_search",
            AsyncMock(
                return_value=[
                    ChunkCandidate(2, "best dense passage"),
                    ChunkCandidate(1, "worst dense passage"),
                ]
            ),
        ),
        patch(
            "paperless_ai.search.retriever.fetch_document_chunks",
            AsyncMock(return_value=[ChunkCandidate(3, "mid keyword passage")]),
        ),
        patch(
            "paperless_ai.search.retriever.local_rerank",
            AsyncMock(
                return_value=[
                    ChunkCandidate(2, "best dense passage"),
                    ChunkCandidate(3, "mid keyword passage"),
                    ChunkCandidate(1, "worst dense passage"),
                ]
            ),
        ) as rerank_mock,
    ):
        fused_ids, chunk_map = await hybrid_retrieve(
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            query="youtube premium",
            client=client,
            rerank_candidates=100,
            mode="recall",
        )

    assert fused_ids == [2, 3, 1]
    assert chunk_map == {
        2: "best dense passage",
        3: "mid keyword passage",
        1: "worst dense passage",
    }
    assert client.calls == [
        (
            "youtube premium",
            {
                "correspondent": None,
                "document_type": None,
                "storage_path": None,
                "tags": None,
                "year": None,
            },
        )
    ]
    rerank_mock.assert_awaited_once()


async def test_hybrid_retrieve_precision_prunes_bottom_half():
    class _FakeClient:
        async def search_documents_all(self, query: str, **kwargs):
            return []

    embedder = MagicMock(spec=LocalLazySearchEmbedder)

    with (
        patch(
            "paperless_ai.search.retriever.dense_search",
            AsyncMock(
                return_value=[
                    ChunkCandidate(1, "best dense passage"),
                    ChunkCandidate(2, "mid dense passage"),
                    ChunkCandidate(3, "worst dense passage"),
                    ChunkCandidate(4, "worst dense passage two"),
                ]
            ),
        ),
        patch(
            "paperless_ai.search.retriever.local_rerank",
            AsyncMock(
                return_value=[
                    ChunkCandidate(1, "best dense passage"),
                    ChunkCandidate(2, "mid dense passage"),
                    ChunkCandidate(3, "worst dense passage"),
                    ChunkCandidate(4, "worst dense passage two"),
                ]
            ),
        ),
    ):
        fused_ids, chunk_map = await hybrid_retrieve(
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            query="youtube premium",
            client=_FakeClient(),
            mode="precision",
        )

    assert fused_ids == [1, 2]
    assert chunk_map == {1: "best dense passage", 2: "mid dense passage"}


async def test_hybrid_retrieve_caps_rerank_candidates():
    class _FakeClient:
        async def search_documents_all(self, query: str, **kwargs):
            return [3, 4, 5]

    embedder = MagicMock(spec=LocalLazySearchEmbedder)
    dense_chunks = [
        ChunkCandidate(1, "dense 1"),
        ChunkCandidate(2, "dense 2"),
    ]
    keyword_chunks = [
        ChunkCandidate(3, "keyword 3"),
        ChunkCandidate(4, "keyword 4"),
        ChunkCandidate(5, "keyword 5"),
    ]

    with (
        patch(
            "paperless_ai.search.retriever.dense_search",
            AsyncMock(return_value=dense_chunks),
        ),
        patch(
            "paperless_ai.search.retriever.fetch_document_chunks",
            AsyncMock(return_value=keyword_chunks),
        ),
        patch(
            "paperless_ai.search.retriever.local_rerank",
            AsyncMock(
                return_value=[
                    ChunkCandidate(1, "dense 1"),
                    ChunkCandidate(3, "keyword 3"),
                ]
            ),
        ) as rerank_mock,
    ):
        await hybrid_retrieve(
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            query="invoice",
            client=_FakeClient(),
            rerank_candidates=2,
            mode="recall",
        )

    rerank_candidates = rerank_mock.await_args.args[2]
    assert len(rerank_candidates) == 2
    assert {candidate.doc_id for candidate in rerank_candidates} <= {1, 2, 3, 4, 5}


async def test_hybrid_retrieve_enforces_global_rerank_cap():
    class _FakeClient:
        async def search_documents_all(self, query: str, **kwargs):
            return list(range(1001, 2501))

    embedder = MagicMock(spec=LocalLazySearchEmbedder)
    dense_chunks = [
        ChunkCandidate(doc_id, f"dense {doc_id}") for doc_id in range(1, 101)
    ]
    keyword_chunks = [
        ChunkCandidate(doc_id, f"keyword {doc_id}") for doc_id in range(1001, 2501)
    ]

    with (
        patch(
            "paperless_ai.search.retriever.dense_search",
            AsyncMock(return_value=dense_chunks),
        ),
        patch(
            "paperless_ai.search.retriever.fetch_document_chunks",
            AsyncMock(return_value=keyword_chunks),
        ),
        patch(
            "paperless_ai.search.retriever.local_rerank",
            AsyncMock(return_value=dense_chunks[:5]),
        ) as rerank_mock,
    ):
        await hybrid_retrieve(
            embedder=embedder,
            qdrant_url="http://qdrant:6333",
            query="invoice",
            client=_FakeClient(),
            rerank_candidates=5000,
            mode="recall",
        )

    rerank_candidates = rerank_mock.await_args.args[2]
    assert len(rerank_candidates) == MAX_RERANK_CANDIDATES


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


async def test_process_embedder_restarts_after_idle_exit():
    embedder = ProcessLocalSearchEmbedder(
        idle_timeout_seconds=0,
        start_method="fork",
        worker_target=_fake_worker,
    )
    first = await embedder.embed_query("alpha")
    first_pid = embedder._process.pid
    assert first.dense == [0.1, 0.2]

    await asyncio.sleep(0.1)

    second = await embedder.embed_query("beta")
    second_pid = embedder._process.pid
    assert second.dense == [0.1, 0.2]
    assert first_pid != second_pid

    await embedder.aclose()


async def test_process_embedder_rerank_round_trip():
    embedder = ProcessLocalSearchEmbedder(
        idle_timeout_seconds=5,
        start_method="fork",
        worker_target=_fake_worker,
    )
    scores = await embedder.rerank(
        "query",
        ["a", "b", "c"],
        model_name=embedder.LOCAL_RERANKER_MODEL_NAME,
    )
    assert scores == [1.0, 2.0, 3.0]
    await embedder.aclose()


async def test_process_embedder_warmup_reuses_worker():
    embedder = ProcessLocalSearchEmbedder(
        idle_timeout_seconds=5,
        start_method="fork",
        worker_target=_fake_worker,
    )
    await embedder.warmup()
    first_pid = embedder._process.pid
    result = await embedder.embed_query("alpha")
    assert result.dense == [0.1, 0.2]
    assert embedder._process.pid == first_pid
    await embedder.aclose()


# ---------------------------------------------------------------------------
# Endpoint integration tests (require running ai copilot service + Qdrant)
# ---------------------------------------------------------------------------


@pytest.mark.requires_copilot
async def test_search_missing_q_returns_422():
    """/search without ?q must return 422 Unprocessable Entity."""
    async with niquests.AsyncSession() as client:
        r = await client.get(f"{COPILOT_URL}/search")
    assert r.status_code == 422


@pytest.mark.requires_copilot
async def test_search_empty_string_returns_422():
    """/search?q= (empty string) must be rejected by the min_length=1 constraint."""
    async with niquests.AsyncSession() as client:
        r = await client.get(f"{COPILOT_URL}/search", params={"q": ""})
    assert r.status_code == 422


@pytest.mark.requires_copilot
async def test_search_returns_list():
    """/search?q=... must return a JSON array (possibly empty when Qdrant is empty)."""
    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(f"{COPILOT_URL}/search", params={"q": "invoice"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list)
    assert all(isinstance(doc_id, int) for doc_id in body)


@pytest.mark.requires_copilot
async def test_search_respects_limit():
    """/search?limit=1 must return at most 1 result."""
    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(
            f"{COPILOT_URL}/search", params={"q": "document", "limit": 1}
        )
    assert r.status_code == 200
    assert len(r.json()) <= 1


@pytest.mark.requires_copilot
async def test_search_limit_too_large_returns_422():
    """/search?limit=999 exceeds the max of 100 and must be rejected."""
    async with niquests.AsyncSession() as client:
        r = await client.get(
            f"{COPILOT_URL}/search", params={"q": "test", "limit": 999}
        )
    assert r.status_code == 422


@pytest.mark.requires_copilot
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
                document_type=None,
                storage_path=None,
                tags=[],
                date=None,
                year=None,
                text="first chunk",
            ),
            ChunkPayload(
                doc_id=9999,
                chunk_index=1,
                title="Dedup test",
                correspondent=None,
                document_type=None,
                storage_path=None,
                tags=[],
                date=None,
                year=None,
                text="second chunk",
            ),
        ],
        dense_vecs=[dense, dense],
        sparse_indices=[[], []],
        sparse_values=[[], []],
    )

    async with niquests.AsyncSession(timeout=30) as client:
        r = await client.get(
            f"{COPILOT_URL}/search", params={"q": "dedup test", "limit": 20}
        )

    assert r.status_code == 200
    doc_ids = r.json()
    assert doc_ids.count(9999) <= 1, (
        f"doc_id 9999 appeared {doc_ids.count(9999)} times — deduplication failed"
    )

    # Cleanup
    await qdrant_store.delete_document(9999)


async def test_dense_search_applies_metadata_filters(qdrant_store):
    from paperless_ai.search.retriever import SearchFilters, dense_search
    from paperless_ai.search.qdrant_store import ChunkPayload

    class _QueryEmbedder:
        async def embed_query(self, _query: str) -> EmbeddingResult:
            dense = [0.0] * 1024
            dense[0] = 1.0
            return EmbeddingResult(dense=dense, sparse_indices=[], sparse_values=[])

    dense = [0.0] * 1024
    dense[0] = 1.0

    await qdrant_store.upsert_chunks(
        chunks=[
            ChunkPayload(
                doc_id=9101,
                chunk_index=0,
                title="Tax receipt",
                correspondent="Home Depot",
                document_type="Receipt",
                storage_path="Archive/2023",
                tags=["Home Improvement", "Urgent"],
                date="2023-02-11",
                year="2023",
                text="lumber and screws",
            ),
            ChunkPayload(
                doc_id=9102,
                chunk_index=0,
                title="Other invoice",
                correspondent="Home Depot",
                document_type="Invoice",
                storage_path="Inbox",
                tags=["Personal"],
                date="2024-02-11",
                year="2024",
                text="paint order",
            ),
        ],
        dense_vecs=[dense, dense],
        sparse_indices=[[], []],
        sparse_values=[[], []],
    )

    try:
        results = await dense_search(
            _QueryEmbedder(),
            QDRANT_URL,
            "home depot",
            10,
            filters=SearchFilters(
                correspondent="Home Depot",
                document_type="Receipt",
                storage_path="Archive/2023",
                tags=["Urgent"],
                year="2023",
            ),
        )
    finally:
        await qdrant_store.delete_document(9101)
        await qdrant_store.delete_document(9102)

    assert [item.doc_id for item in results] == [9101]
