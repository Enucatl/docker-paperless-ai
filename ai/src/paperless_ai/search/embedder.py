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

import asyncio
import logging
import time
from dataclasses import dataclass, field

import niquests

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
        self._client = niquests.AsyncSession(timeout=120)

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


class LocalLazySearchEmbedder:
    """CPU-based embedder using FastEmbed for ad-hoc search queries.

    The model is loaded on first use and automatically unloaded after a
    configurable idle period, so it consumes no RAM when the search API
    is not being used ("scale-to-zero" for RAM).
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self) -> None:
        self.model = None  # fastembed.TextEmbedding | None
        self._last_used: float = 0.0

    def _get_model(self):
        """Return the FastEmbed model, loading it on first call."""
        if self.model is None:
            from fastembed import TextEmbedding

            log.info("Loading FastEmbed %s into RAM…", self.MODEL_NAME)
            self.model = TextEmbedding(model_name=self.MODEL_NAME)
        self._last_used = time.monotonic()
        return self.model

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single search query.

        FastEmbed is CPU-bound; we run it in a thread pool to avoid
        blocking the FastAPI event loop.
        """

        def _embed() -> list[float]:
            model = self._get_model()
            return list(model.embed([query]))[0].tolist()

        dense = await asyncio.to_thread(_embed)
        return EmbeddingResult(dense=dense)

    async def idle_watcher(self, timeout_seconds: int = 300) -> None:
        """Background task: unload the model after it has been idle."""
        while True:
            await asyncio.sleep(60)
            if self.model is not None and (time.monotonic() - self._last_used) > timeout_seconds:
                log.info(
                    "FastEmbed model idle for >%ds — freeing RAM", timeout_seconds
                )
                self.model = None
                import gc

                gc.collect()
