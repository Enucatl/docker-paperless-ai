"""
Hybrid retrieval system: dense + keyword + RRF fusion + local BGE reranking.

Chat retrieval differs from the HTTP endpoint in two ways:
  - chat can choose between precision and recall presets
  - chat reranks chunk candidates before deduplicating to documents
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from paperless_ai.core.telemetry import set_span_attributes, start_span
from paperless_ai.search.embedder_types import SearchEmbedder
from paperless_ai.search.qdrant_store import COLLECTION

log = logging.getLogger(__name__)

K = 25
N = 50
MAX_RERANK_CANDIDATES = 1000
RRF_K = 60
RETRIEVAL_MODE_DENSE_K = {"precision": 40, "recall": 100}
PRECISION_PRUNE_FRACTION = 0.50

RetrievalMode = Literal["precision", "recall"]


@dataclass
class ChunkCandidate:
    doc_id: int
    chunk_text: str


@dataclass
class SearchFilters:
    correspondent: Optional[str] = None
    document_type: Optional[str] = None
    storage_path: Optional[str] = None
    tags: Optional[list[str]] = None
    year: Optional[str] = None


def build_qdrant_filter(filters: SearchFilters) -> Filter | None:
    """Build a Qdrant filter from human-readable metadata values."""
    must_conditions: list[FieldCondition] = []
    if filters.correspondent:
        must_conditions.append(
            FieldCondition(
                key="correspondent",
                match=MatchValue(value=filters.correspondent),
            )
        )
    if filters.document_type:
        must_conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchValue(value=filters.document_type),
            )
        )
    if filters.storage_path:
        must_conditions.append(
            FieldCondition(
                key="storage_path",
                match=MatchValue(value=filters.storage_path),
            )
        )
    if filters.tags:
        must_conditions.append(
            FieldCondition(key="tags", match=MatchAny(any=filters.tags))
        )
    if filters.year:
        must_conditions.append(
            FieldCondition(key="year", match=MatchValue(value=str(filters.year)))
        )
    return Filter(must=must_conditions) if must_conditions else None


def _extract_qdrant_hits(response) -> list:
    """Normalize AsyncQdrantClient query responses across client versions."""
    if hasattr(response, "points"):
        return list(response.points)
    if isinstance(response, tuple):
        return list(response[0])
    return list(response)


def _doc_ids_from_chunks(chunks: list[ChunkCandidate]) -> list[int]:
    seen: set[int] = set()
    doc_ids: list[int] = []
    for chunk in chunks:
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        doc_ids.append(chunk.doc_id)
    return doc_ids


def _chunk_map_from_chunks(chunks: list[ChunkCandidate]) -> dict[int, str]:
    chunk_map: dict[int, str] = {}
    for chunk in chunks:
        chunk_map.setdefault(chunk.doc_id, chunk.chunk_text)
    return chunk_map


async def dense_search(
    embedder: SearchEmbedder,
    qdrant_url: str,
    query: str,
    k: int,
    filters: SearchFilters | None = None,
    *,
    qdrant_client: AsyncQdrantClient | None = None,
) -> list[ChunkCandidate]:
    """Search by dense similarity and return ranked chunk candidates."""
    with start_span(
        "paperless_ai.search.dense_search",
        **{
            "paperless_ai.search.query": query,
            "paperless_ai.search.dense_k": k,
        },
    ) as span:
        result = await embedder.embed_query(query)
        qdrant = qdrant_client or AsyncQdrantClient(url=qdrant_url)
        try:
            response = await qdrant.query_points(
                collection_name=COLLECTION,
                query=result.dense,
                using="dense",
                limit=k,
                with_payload=True,
                query_filter=build_qdrant_filter(filters or SearchFilters()),
            )
        finally:
            if qdrant_client is None:
                await qdrant.close()

        hits = _extract_qdrant_hits(response)
        candidates: list[ChunkCandidate] = []
        for hit in hits:
            if not hit.payload or hit.payload.get("doc_id") is None:
                continue
            candidates.append(
                ChunkCandidate(
                    doc_id=int(hit.payload["doc_id"]),
                    chunk_text=str(hit.payload.get("text") or ""),
                )
            )
        set_span_attributes(
            span,
            **{
                "paperless_ai.search.dense_chunk_count": len(candidates),
                "paperless_ai.search.dense_doc_count": len(
                    _doc_ids_from_chunks(candidates)
                ),
            },
        )
        return candidates


def rrf_fuse(
    dense_ids: list[int],
    keyword_ids: list[int],
    k: int = 60,
) -> list[int]:
    """Reciprocal Rank Fusion: combine dense and keyword rankings."""
    scores: defaultdict[int, float] = defaultdict(float)
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] += 1 / (k + rank)
    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores[doc_id] += 1 / (k + rank)
    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


async def fetch_document_chunks(
    qdrant_url: str,
    doc_ids: list[int],
    filters: SearchFilters | None = None,
    *,
    batch_size: int = 64,
    page_size: int = 256,
    qdrant_client: AsyncQdrantClient | None = None,
) -> list[ChunkCandidate]:
    """Fetch all known Qdrant chunks for the provided document IDs."""
    if not doc_ids:
        return []
    with start_span(
        "paperless_ai.search.fetch_document_chunks",
        **{
            "paperless_ai.search.doc_id_count": len(doc_ids),
        },
    ) as span:
        qdrant = qdrant_client or AsyncQdrantClient(url=qdrant_url)
        try:
            candidates: list[ChunkCandidate] = []
            doc_filter = build_qdrant_filter(filters or SearchFilters())
            for start in range(0, len(doc_ids), batch_size):
                batch = doc_ids[start : start + batch_size]
                offset = None
                while True:
                    must_conditions = [
                        FieldCondition(key="doc_id", match=MatchAny(any=batch)),
                    ]
                    if doc_filter is not None:
                        must_conditions.extend(doc_filter.must or [])
                    response = await qdrant.scroll(
                        collection_name=COLLECTION,
                        scroll_filter=Filter(must=must_conditions),
                        with_payload=True,
                        limit=page_size,
                        offset=offset,
                    )
                    points, offset = response
                    for point in points:
                        if not point.payload or point.payload.get("doc_id") is None:
                            continue
                        candidates.append(
                            ChunkCandidate(
                                doc_id=int(point.payload["doc_id"]),
                                chunk_text=str(point.payload.get("text") or ""),
                            )
                        )
                    if offset is None:
                        break
            set_span_attributes(
                span,
                **{
                    "paperless_ai.search.keyword_chunk_count": len(candidates),
                },
            )
            return candidates
        finally:
            if qdrant_client is None:
                await qdrant.close()


def _order_chunk_candidates(
    fused_doc_ids: list[int],
    dense_chunks: list[ChunkCandidate],
    keyword_chunks: list[ChunkCandidate],
) -> list[ChunkCandidate]:
    doc_rank = {doc_id: idx for idx, doc_id in enumerate(fused_doc_ids)}
    ordered: list[ChunkCandidate] = []
    for candidate in [*dense_chunks, *keyword_chunks]:
        if candidate.doc_id in doc_rank:
            ordered.append(candidate)
    ordered.sort(key=lambda candidate: doc_rank[candidate.doc_id])
    return ordered


async def local_rerank(
    embedder: SearchEmbedder,
    query: str,
    candidates: list[ChunkCandidate],
) -> list[ChunkCandidate]:
    """Rerank chunk candidates with the local BGE cross-encoder."""
    if not candidates:
        return []
    with start_span(
        "paperless_ai.search.local_rerank",
        **{
            "paperless_ai.search.query": query,
            "paperless_ai.search.chunk_candidate_count": len(candidates),
        },
    ) as span:
        passages = [candidate.chunk_text or "" for candidate in candidates]
        try:
            scores = await embedder.rerank(
                query,
                passages,
                model_name=embedder.LOCAL_RERANKER_MODEL_NAME,
            )
        except Exception as exc:
            log.warning("Local rerank failed: %s", exc)
            set_span_attributes(
                span, **{"paperless_ai.search.local_rerank_failed": True}
            )
            return candidates
        ranked_indices = sorted(
            range(len(candidates)), key=lambda idx: scores[idx], reverse=True
        )
        reranked = [candidates[idx] for idx in ranked_indices]
        set_span_attributes(
            span, **{"paperless_ai.search.local_rerank_output_count": len(reranked)}
        )
        return reranked


def _prune_precision_chunks(candidates: list[ChunkCandidate]) -> list[ChunkCandidate]:
    if len(candidates) <= 1:
        return candidates
    keep_count = max(1, int(len(candidates) * (1.0 - PRECISION_PRUNE_FRACTION)))
    return candidates[:keep_count]


def _dedupe_documents(
    candidates: list[ChunkCandidate],
) -> tuple[list[int], dict[int, str]]:
    doc_ids: list[int] = []
    chunk_map: dict[int, str] = {}
    seen: set[int] = set()
    for candidate in candidates:
        if candidate.doc_id in seen:
            continue
        seen.add(candidate.doc_id)
        doc_ids.append(candidate.doc_id)
        chunk_map[candidate.doc_id] = candidate.chunk_text
    return doc_ids, chunk_map


async def hybrid_retrieve(
    *,
    embedder: SearchEmbedder,
    qdrant_url: str,
    query: str,
    client=None,
    filters: SearchFilters | None = None,
    dense_k: int = K,
    rerank_candidates: int = N,
    rrf_k: int = RRF_K,
    mode: RetrievalMode = "precision",
    qdrant_client: AsyncQdrantClient | None = None,
) -> tuple[list[int], dict[int, str]]:
    """Shared hybrid retrieval pipeline for chat and the HTTP search endpoint."""
    with start_span(
        "paperless_ai.search.hybrid_retrieve",
        **{
            "paperless_ai.search.query": query,
            "paperless_ai.search.mode": mode,
            "paperless_ai.search.dense_k": dense_k,
            "paperless_ai.search.rrf_k": rrf_k,
        },
    ) as span:
        resolved_filters = filters or SearchFilters()

        keyword_coro = (
            asyncio.sleep(0, result=[])
            if client is None
            else client.search_documents_all(
                query,
                correspondent=resolved_filters.correspondent,
                document_type=resolved_filters.document_type,
                storage_path=resolved_filters.storage_path,
                tags=resolved_filters.tags,
                year=resolved_filters.year,
            )
        )

        dense_result, keyword_result = await asyncio.gather(
            dense_search(
                embedder,
                qdrant_url,
                query,
                dense_k,
                filters=resolved_filters,
                qdrant_client=qdrant_client,
            ),
            keyword_coro,
            return_exceptions=True,
        )
        if isinstance(dense_result, BaseException):
            raise dense_result

        dense_chunks: list[ChunkCandidate] = dense_result
        keyword_ids: list[int] = (
            keyword_result if not isinstance(keyword_result, BaseException) else []
        )
        if isinstance(keyword_result, BaseException):
            set_span_attributes(span, **{"paperless_ai.search.keyword_failed": True})

        dense_ids = _doc_ids_from_chunks(dense_chunks)
        fused_ids = (
            rrf_fuse(dense_ids, keyword_ids, k=rrf_k) if keyword_ids else dense_ids
        )
        set_span_attributes(
            span,
            **{
                "paperless_ai.search.dense_doc_count": len(dense_ids),
                "paperless_ai.search.keyword_doc_count": len(keyword_ids),
                "paperless_ai.search.fused_doc_count": len(fused_ids),
            },
        )
        if not fused_ids:
            return [], {}

        keyword_only_doc_ids = [
            doc_id for doc_id in fused_ids if doc_id not in set(dense_ids)
        ]
        keyword_chunks = await fetch_document_chunks(
            qdrant_url,
            keyword_only_doc_ids,
            filters=resolved_filters,
            qdrant_client=qdrant_client,
        )
        chunk_candidates = _order_chunk_candidates(
            fused_ids, dense_chunks, keyword_chunks
        )
        bounded_rerank_candidates = min(
            MAX_RERANK_CANDIDATES, max(1, rerank_candidates)
        )
        bounded_chunk_candidates = chunk_candidates[:bounded_rerank_candidates]
        set_span_attributes(
            span,
            **{
                "paperless_ai.search.chunk_candidate_count": len(chunk_candidates),
                "paperless_ai.search.rerank_candidate_limit": bounded_rerank_candidates,
                "paperless_ai.search.rerank_candidate_count": len(
                    bounded_chunk_candidates
                ),
            },
        )
        if not bounded_chunk_candidates:
            return fused_ids, {}

        reranked_chunks = await local_rerank(embedder, query, bounded_chunk_candidates)
        if mode == "precision":
            before_prune = len(reranked_chunks)
            reranked_chunks = _prune_precision_chunks(reranked_chunks)
            set_span_attributes(
                span,
                **{
                    "paperless_ai.search.precision_pruned_count": before_prune
                    - len(reranked_chunks),
                    "paperless_ai.search.post_prune_chunk_count": len(reranked_chunks),
                },
            )
        doc_ids, chunk_map = _dedupe_documents(reranked_chunks)
        set_span_attributes(
            span,
            **{
                "paperless_ai.search.final_doc_count": len(doc_ids),
            },
        )
        return doc_ids, chunk_map
