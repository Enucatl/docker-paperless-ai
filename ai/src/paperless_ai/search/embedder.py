"""
Async client for the Infinity embedding server (michaelfeil/infinity).

Infinity extends the OpenAI /embeddings spec to return *both* a dense vector
and a sparse (BM25/lexical) vector per input text in a single call, making it
ideal for hybrid search with bge-m3.

Expected response shape:
    {
        "data": [
            {
                "embedding": [0.01, -0.23, ...],       # dense 1024-d
                "sparse_embedding": {
                    "indices": [42, 1337, ...],
                    "values":  [0.71, 0.33, ...]
                }
            },
            ...
        ]
    }
"""

import logging
from dataclasses import dataclass, field

import httpx

log = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)


class InfinityEmbedder:
    def __init__(
        self,
        base_url: str = "http://localhost:8102",
        model: str = "BAAI/bge-m3",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=120)

    async def aclose(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed *texts* and return dense + sparse vectors for each."""
        r = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"input": texts, "model": self._model},
        )
        r.raise_for_status()

        results = []
        for item in r.json()["data"]:
            dense = item.get("embedding", [])
            sparse = item.get("sparse_embedding") or {}
            results.append(
                EmbeddingResult(
                    dense=dense,
                    sparse_indices=sparse.get("indices", []),
                    sparse_values=sparse.get("values", []),
                )
            )
        return results

    async def check_connectivity(self) -> bool:
        """Return True if the Infinity server is reachable."""
        try:
            r = await self._client.get(f"{self._base_url}/health", timeout=5)
            return r.is_success
        except Exception:
            return False
