"""Lightweight types and interfaces for local search inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)


class SearchEmbedder(Protocol):
    LOCAL_RERANKER_MODEL_NAME: str

    async def embed_query(self, query: str) -> EmbeddingResult: ...

    async def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        model_name: str,
        normalize: bool = False,
    ) -> list[float]: ...
