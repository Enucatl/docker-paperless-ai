"""
Hybrid retrieval system: dense + keyword + RRF fusion + LLM reranking.

Two-Tower architecture:
  Track A: Dense embeddings (FastEmbed query → Qdrant cosine) → top K chunks → rolled to doc_ids
  Track B: Keyword search (Paperless full-text API)
  Merge:   Reciprocal Rank Fusion (RRF) to combine incompatible score scales
  Rerank:  LLM-as-a-Judge (litellm) to filter false positives and reorder by semantic relevance
"""

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import litellm
import niquests
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.qdrant_store import COLLECTION

log = logging.getLogger(__name__)

K = 25
N = 50
RRF_K = 60


@dataclass
class ScoredDoc:
    """Document with RRF score and optional chunk text for reranking."""
    doc_id: int
    rrf_score: float
    chunk_text: Optional[str] = None


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


async def dense_search(
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
    query: str,
    k: int,
    filters: SearchFilters | None = None,
) -> list[tuple[int, str]]:
    """
    Search by dense similarity: embed query → Qdrant cosine → roll up to doc_ids.

    Args:
        embedder: LocalLazySearchEmbedder (CPU FastEmbed)
        qdrant_url: Qdrant base URL
        query: user search query
        k: max chunks to retrieve

    Returns:
        list of (doc_id, chunk_text) in rank order, deduplicated by highest chunk score
    """
    result = await embedder.embed_query(query)

    qdrant = AsyncQdrantClient(url=qdrant_url)
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
        await qdrant.close()
    hits = _extract_qdrant_hits(response)

    # Roll up chunks to doc_ids: keep highest chunk score per doc
    seen: dict[int, str] = {}
    for hit in hits:
        doc_id = hit.payload.get("doc_id") if hit.payload else None
        text = hit.payload.get("text") if hit.payload else None
        if doc_id is not None and doc_id not in seen:
            seen[doc_id] = text or ""

    return [(doc_id, text) for doc_id, text in seen.items()]


async def keyword_search(
    paperless_url: str,
    token: str,
    query: str,
    page_size: int = 50,
) -> list[int]:
    """
    Search via Paperless-ngx full-text API.

    Args:
        paperless_url: Paperless base URL
        token: API token
        query: user search query
        page_size: max results to fetch

    Returns:
        list of doc_ids in Paperless relevance order

    Raises:
        niquests.HTTPError: on API errors
    """
    async with niquests.AsyncSession(
        base_url=paperless_url,
        headers={"Authorization": f"Token {token}"},
        timeout=60,
    ) as session:
        r = await session.get(
            "/api/documents/",
            params={"query": query, "page_size": page_size, "fields": "id"},
        )
        r.raise_for_status()
        return [doc["id"] for doc in r.json().get("results", [])]


def rrf_fuse(
    dense_ids: list[int],
    keyword_ids: list[int],
    k: int = 60,
) -> list[int]:
    """
    Reciprocal Rank Fusion: combine dense and keyword rankings.

    Score = 1/(k + rank_dense) + 1/(k + rank_keyword)
    A doc appearing in both lists scores higher than appearing in one.

    Args:
        dense_ids: ordered list from dense search
        keyword_ids: ordered list from keyword search
        k: smoothing constant (default 60, per RRF literature)

    Returns:
        merged doc_ids sorted descending by RRF score
    """
    scores: defaultdict[int, float] = defaultdict(float)

    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] += 1 / (k + rank)

    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores[doc_id] += 1 / (k + rank)

    # Sort descending by score; ties broken by appearance order (stable sort)
    return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)


async def llm_rerank(
    query: str,
    candidates: list[ScoredDoc],
    model: str,
    api_base: Optional[str],
    top_n: int,
) -> list[int]:
    """
    Rerank top N candidates using LLM-as-a-Judge (litellm).

    Concurrently evaluates each candidate for relevance and assigns a score (1-10).
    Filters out docs marked as irrelevant, sorts by score.

    Args:
        query: user search query
        candidates: list of ScoredDoc with chunk_text
        model: LiteLLM model string (e.g., "gemini/gemini-2.5-flash")
        api_base: optional override for model API endpoint
        top_n: max candidates to evaluate

    Returns:
        list of doc_ids sorted by LLM relevance score (highest first)
        If all evaluations fail, returns input order as fallback
    """
    if not candidates:
        return []

    candidates = candidates[:top_n]

    async def judge_candidate(cand: ScoredDoc) -> tuple[int, Optional[dict]]:
        """Judge a single candidate. Returns (doc_id, {is_relevant, score, reason} or None on error)."""
        prompt = f"""You are a relevance judge. Given a search query and a document excerpt, determine if the document is relevant to the query.

Search query: {query}

Document (ID {cand.doc_id}):
{cand.chunk_text or "(no text available)"}

Output a JSON object with exactly these fields:
- is_relevant: boolean
- score: integer 1-10 (higher = more relevant)
- reason: string explaining your judgment

Respond ONLY with valid JSON, no other text."""

        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 200,
        }
        if api_base:
            kwargs["api_base"] = api_base

        try:
            resp = await litellm.acompletion(**kwargs)
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            return (cand.doc_id, data)
        except Exception as e:
            log.warning("LLM rerank failed for doc %d: %s", cand.doc_id, e)
            return (cand.doc_id, None)

    # Concurrent evaluation
    tasks = [judge_candidate(cand) for cand in candidates]
    results = await asyncio.gather(*tasks)

    # Filter and sort
    scored: list[tuple[int, int]] = []
    for doc_id, judgment in results:
        if judgment and judgment.get("is_relevant"):
            score = judgment.get("score", 0)
            if isinstance(score, int) and 1 <= score <= 10:
                scored.append((doc_id, score))

    # Sort descending by score; ties broken by input order (stable)
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        return [doc_id for doc_id, _ in scored]

    # Fallback: no relevant docs found, return input order
    log.debug("No docs judged as relevant; returning input order")
    return [cand.doc_id for cand in candidates]


async def hybrid_retrieve(
    *,
    embedder: LocalLazySearchEmbedder,
    qdrant_url: str,
    query: str,
    client=None,
    filters: SearchFilters | None = None,
    rerank_model: str | None = None,
    rerank_api_base: str | None = None,
    dense_k: int = K,
    rerank_candidates: int = N,
    rrf_k: int = RRF_K,
) -> tuple[list[int], dict[int, str]]:
    """Shared hybrid retrieval pipeline for chat and the HTTP search endpoint."""
    resolved_filters = filters or SearchFilters()
    has_metadata_filters = any(
        [
            resolved_filters.correspondent,
            resolved_filters.document_type,
            resolved_filters.storage_path,
            resolved_filters.tags,
            resolved_filters.year,
        ]
    )

    keyword_coro = (
        asyncio.sleep(0, result=[])
        if has_metadata_filters or client is None
        else client.search_documents(query, page_size=rerank_candidates)
    )

    dense_result, keyword_result = await asyncio.gather(
        dense_search(embedder, qdrant_url, query, dense_k, filters=resolved_filters),
        keyword_coro,
        return_exceptions=True,
    )

    if isinstance(dense_result, BaseException):
        raise dense_result

    dense_results: list[tuple[int, str]] = dense_result
    keyword_ids: list[int] = keyword_result if not isinstance(keyword_result, BaseException) else []

    dense_ids = [doc_id for doc_id, _ in dense_results]
    chunk_map = {doc_id: text for doc_id, text in dense_results}

    if keyword_ids:
        fused_ids = rrf_fuse(dense_ids, keyword_ids, k=rrf_k)
    else:
        fused_ids = dense_ids

    if rerank_model and fused_ids:
        candidates = [
            ScoredDoc(doc_id, 0.0, chunk_map.get(doc_id))
            for doc_id in fused_ids[:rerank_candidates]
        ]
        fused_ids = await llm_rerank(
            query,
            candidates,
            rerank_model,
            rerank_api_base,
            rerank_candidates,
        )

    return fused_ids, chunk_map
