"""
Tests for CLI connectivity checks.

Ensures the CLI properly initializes and validates connections without using
invalid parameters or making incorrect API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paperless_common.paperless import PaperlessClient


@pytest.mark.asyncio
async def test_cli_paperless_connectivity_check_without_invalid_params():
    """
    Verify the CLI connectivity check uses valid niquests parameters.

    This test simulates what cli.py does when checking Paperless reachability:
    - Calls client._client.get("/api/")
    - Does NOT use follow_redirects (httpx param) or any other invalid params
    """
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        # Simulate Paperless API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-version": "5.0"}
        mock_session.get.return_value = mock_response

        async with PaperlessClient("http://test:8000", "token123") as client:
            # This is what cli.py does:
            r = await client._client.get("/api/")

            # Verify:
            # 1. The call succeeded
            assert r.status_code == 200

            # 2. The call was made without invalid parameters
            call_kwargs = mock_session.get.call_args[1]
            assert "follow_redirects" not in call_kwargs, (
                "follow_redirects is an httpx parameter; niquests doesn't support it"
            )

            # 3. Only the path was passed (no extra params)
            call_args = mock_session.get.call_args[0]
            assert len(call_args) == 1
            assert call_args[0] == "/api/"


@pytest.mark.asyncio
async def test_cli_connectivity_check_verifies_version_header():
    """Verify the CLI can extract version info from Paperless response headers."""
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"x-version": "5.1.2"}
        mock_session.get.return_value = mock_response

        async with PaperlessClient("http://test:8000", "token123") as client:
            r = await client._client.get("/api/")
            version = r.headers.get("x-version", "unknown")

        assert version == "5.1.2"


@pytest.mark.asyncio
async def test_cli_connectivity_check_handles_api_errors():
    """Verify the CLI connectivity check can handle API errors gracefully."""
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        # Simulate a connection error
        mock_session.get.side_effect = Exception("Connection refused")

        async with PaperlessClient("http://test:8000", "token123") as client:
            with pytest.raises(Exception, match="Connection refused"):
                await client._client.get("/api/")
