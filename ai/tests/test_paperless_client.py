"""
Tests for PaperlessClient async context manager and API compatibility.

Ensures niquests AsyncSession is used correctly (e.g., close() not aclose(),
no follow_redirects parameter, etc).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paperless_ai.core.paperless import PaperlessClient


@pytest.mark.asyncio
async def test_paperless_client_context_manager():
    """Verify PaperlessClient async context manager properly calls close() on exit."""
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        # Make sure close() exists and is callable
        mock_session.close = AsyncMock()

        # Create client and verify context manager calls close on exit
        async with PaperlessClient("http://test:8000", "token123") as client:
            assert client._client == mock_session

        # Verify close() was called when exiting context (not aclose())
        mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_paperless_client_api_calls_use_correct_parameters():
    """Verify API calls don't use httpx-specific parameters like follow_redirects."""
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json = MagicMock(return_value={"results": []})
        mock_session.get = AsyncMock(return_value=mock_response)

        async with PaperlessClient("http://test:8000", "token123") as client:
            try:
                await client.get_tag_id("test", create=False)
            except ValueError:
                # Expected when tag not found
                pass

        # Verify get() was called without follow_redirects
        call_kwargs = mock_session.get.call_args[1]
        assert "follow_redirects" not in call_kwargs, \
            "niquests.AsyncSession doesn't support follow_redirects; use allow_redirects instead"


@pytest.mark.asyncio
async def test_paperless_client_aclose_method_exists():
    """Verify PaperlessClient.aclose() method exists and delegates to session.close()."""
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        client = PaperlessClient("http://test:8000", "token123")
        await client.aclose()

        # Verify close() (not aclose()) was called on the session
        mock_session.close.assert_called_once()
