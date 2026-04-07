"""
Async client for an embeddings API.

The batch embed worker uses an OpenAI-compatible embeddings endpoint, such as
vLLM's `/v1/embeddings`. Some backends also return an optional
`sparse_embedding` extension; when absent, sparse vectors are left empty.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import litellm
import niquests

log = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    dense: list[float]
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)


class EmbeddingAPIEmbedder:
    def __init__(
        self,
        base_url: str = "http://localhost:8102",
        model: str = "BAAI/bge-large-en-v1.5",
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = niquests.AsyncSession(timeout=120)

    async def aclose(self) -> None:
        """Close the underlying HTTP client connection pool."""
        await self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed *texts* and return dense vectors plus optional sparse vectors."""
        response = await litellm.aembedding(
            model=self._model,
            input=texts,
            api_base=f"{self._base_url}/v1",
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            custom_llm_provider="openai",
        )

        results = []
        for item in response.data:
            item_dict = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            dense = item_dict.get("embedding", [])
            sparse = item_dict.get("sparse_embedding") or {}
            results.append(
                EmbeddingResult(
                    dense=dense,
                    sparse_indices=sparse.get("indices", []),
                    sparse_values=sparse.get("values", []),
                )
            )
        return results

    async def check_connectivity(self) -> bool:
        """Return True if the embeddings server is reachable."""
        for path in ("/health", "/v1/models", "/models"):
            try:
                r = await self._client.get(f"{self._base_url}{path}", timeout=5)
                if getattr(r, "ok", False):
                    return True
            except Exception:
                continue
        return False


class LocalLazySearchEmbedder:
    """CPU-based embedder using SentenceTransformers for ad-hoc search queries.

    The model is loaded on first use and automatically unloaded after a
    configurable idle period, so it consumes no RAM when the search API
    is not being used ("scale-to-zero" for RAM).
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self) -> None:
        self.model = None  # sentence_transformers.SentenceTransformer | None
        self._lock = threading.Lock()
        self._last_used: float = 0.0

    def _get_model(self):
        """Return the SentenceTransformer model, loading it on first call.

        Uses double-checked locking so concurrent threads don't each load the
        1 GB+ model into RAM simultaneously.
        """
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("Loading SentenceTransformer %s into RAM…", self.MODEL_NAME)
                    self.model = SentenceTransformer(self.MODEL_NAME, trust_remote_code=True)
        self._last_used = time.monotonic()
        return self.model

    async def embed_query(self, query: str) -> EmbeddingResult:
        """Embed a single search query.

        SentenceTransformer is CPU-bound; we run it in a thread pool to avoid
        blocking the FastAPI event loop.
        """

        def _embed() -> list[float]:
            model = self._get_model()
            return model.encode(query).tolist()

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

                await asyncio.to_thread(gc.collect)
