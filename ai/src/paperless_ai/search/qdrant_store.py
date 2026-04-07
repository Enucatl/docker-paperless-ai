"""
Qdrant vector store for document chunks.

Collection schema:
  - Named dense vector  "dense"  : 1 024 dimensions, cosine distance (bge-m3)
  - Named sparse vector "sparse" : BM25/lexical weights (bge-m3 sparse head)

Point ID scheme: doc_id * 10_000 + chunk_index
  - Deterministic → upserts are idempotent
  - Allows efficient deletion of all chunks for a document via payload filter

Payload per point:
  {
    "doc_id":       int,
    "chunk_index":  int,
    "title":        str | null,
    "correspondent":str | null,
    "date":         str | null,   # YYYY-MM-DD
    "text":         str,
  }
"""

import logging
from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

log = logging.getLogger(__name__)

COLLECTION = "paperless_documents"
DENSE_DIM = 1024


@dataclass
class ChunkPayload:
    doc_id: int
    chunk_index: int
    title: Optional[str]
    correspondent: Optional[str]
    date: Optional[str]
    text: str


class QdrantDocumentStore:
    def __init__(self, url: str = "http://qdrant:6333"):
        self._client = AsyncQdrantClient(url=url)

    async def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if COLLECTION not in names:
            await self._client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams())
                },
            )
            log.info("Created Qdrant collection '%s'", COLLECTION)
        else:
            log.debug("Qdrant collection '%s' already exists", COLLECTION)

    async def upsert_chunks(
        self,
        chunks: list[ChunkPayload],
        dense_vecs: list[list[float]],
        sparse_indices: list[list[int]],
        sparse_values: list[list[float]],
    ) -> None:
        """Upsert all chunks for a document into Qdrant."""
        points = [
            PointStruct(
                id=chunk.doc_id * 10_000 + chunk.chunk_index,
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=s_idx, values=s_val),
                },
                payload={
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "correspondent": chunk.correspondent,
                    "date": chunk.date,
                    "text": chunk.text,
                },
            )
            for chunk, dense, s_idx, s_val in zip(
                chunks, dense_vecs, sparse_indices, sparse_values
            )
        ]
        if points:
            await self._client.upsert(collection_name=COLLECTION, points=points)
            log.debug("Upserted %d chunk(s) for doc %d", len(points), chunks[0].doc_id)

    async def delete_document(self, doc_id: int) -> None:
        """Delete all vectors for a document (called before re-embedding on update)."""
        await self._client.delete(
            collection_name=COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
        log.debug("Deleted vectors for doc %d", doc_id)
