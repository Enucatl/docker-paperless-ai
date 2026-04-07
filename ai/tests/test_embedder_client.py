"""
Tests for EmbeddingAPIEmbedder async context manager and API compatibility.

Ensures niquests AsyncSession is used correctly (e.g., close() not aclose()).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paperless_ai.search.embedder import EmbeddingAPIEmbedder


@pytest.mark.asyncio
async def test_embedding_api_embedder_context_manager():
    """Verify EmbeddingAPIEmbedder async context manager works (uses close() not aclose())."""
    with patch("paperless_ai.search.embedder.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        # Simulate a working embedding response
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_response.json.return_value = {
            "data": [
                {
                    "embedding": [0.1] * 1024,
                    "sparse_embedding": {"indices": [1, 2], "values": [0.5, 0.3]},
                }
            ]
        }
        mock_session.post.return_value = mock_response

        async with EmbeddingAPIEmbedder("http://test:8102", "BAAI/bge-m3") as embedder:
            results = await embedder.embed(["test text"])
            assert len(results) == 1
            mock_session.post.assert_called_once_with(
                "http://test:8102/v1/embeddings",
                json={"input": ["test text"], "model": "BAAI/bge-m3"},
            )

        # Verify close() was called (not aclose())
        mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_embedding_api_embedder_aclose_uses_close():
    """Verify EmbeddingAPIEmbedder.aclose() delegates to session.close()."""
    with patch("paperless_ai.search.embedder.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        embedder = EmbeddingAPIEmbedder("http://test:8102", "BAAI/bge-m3")
        await embedder.aclose()

        # Verify close() (not aclose()) was called on the session
        mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_embedding_api_embedder_health_check():
    """Verify EmbeddingAPIEmbedder.check_connectivity() works with niquests."""
    with patch("paperless_ai.search.embedder.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.is_success = True
        mock_session.get.return_value = mock_response

        embedder = EmbeddingAPIEmbedder("http://test:8102", "BAAI/bge-m3")
        is_healthy = await embedder.check_connectivity()

        assert is_healthy is True
        # Verify get() was called without invalid parameters
        mock_session.get.assert_called_once()
        call_args, call_kwargs = mock_session.get.call_args
        assert call_args[0] == "http://test:8102/health"
        assert "follow_redirects" not in call_kwargs
        assert "allow_redirects" not in call_kwargs or call_kwargs["timeout"] == 5
