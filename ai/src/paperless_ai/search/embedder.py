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

import litellm
import niquests

from paperless_common.telemetry import add_litellm_metadata
from paperless_ai.search.embedder_types import EmbeddingResult
from paperless_ai.search.flag_reranker import FlagReranker

log = logging.getLogger(__name__)


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
        kwargs = {
            "model": self._model,
            "input": texts,
            "api_base": f"{self._base_url}/v1",
            "api_key": os.environ.get("OPENAI_API_KEY", "dummy"),
            "custom_llm_provider": "openai",
            "encoding_format": "float",
        }
        add_litellm_metadata(
            kwargs,
            stage="embedding",
            operation="embed_document_chunks",
        )
        response = await litellm.aembedding(**kwargs)

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
    LOCAL_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

    def __init__(self) -> None:
        self.model = None  # sentence_transformers.SentenceTransformer | None
        self._reranker = None
        self._reranker_model_name: str | None = None
        self._lock = threading.Lock()

    def _get_model(self):
        """Return the SentenceTransformer model, loading it on first call.

        Uses double-checked locking so concurrent threads don't each load the
        1 GB+ model into RAM simultaneously.
        """
        if self.model is None:
            with self._lock:
                if self.model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info(
                        "Loading SentenceTransformer %s into RAM…", self.MODEL_NAME
                    )
                    self.model = SentenceTransformer(
                        self.MODEL_NAME, trust_remote_code=True
                    )
        return self.model

    def _get_reranker(self, model_name: str):
        """Return the local FlagEmbedding reranker, loading on first call."""
        if self._reranker is None or self._reranker_model_name != model_name:
            with self._lock:
                if self._reranker is None or self._reranker_model_name != model_name:
                    import torch

                    log.info("Loading local reranker %s into RAM…", model_name)
                    self._reranker = FlagReranker(
                        model_name,
                        use_fp16=torch.cuda.is_available(),
                    )
                    self._reranker_model_name = model_name
        return self._reranker

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

    async def rerank(
        self,
        query: str,
        passages: list[str],
        *,
        model_name: str,
        normalize: bool = False,
    ) -> list[float]:
        """Score query/passage pairs with the local FlagEmbedding reranker."""

        def _score() -> list[float]:
            reranker = self._get_reranker(model_name)
            pairs = [[query, passage] for passage in passages]
            scores = reranker.compute_score(pairs, normalize=normalize)
            if isinstance(scores, float):
                return [scores]
            return list(scores)

        return await asyncio.to_thread(_score)
