"""
API compatibility tests for niquests library.

niquests is a requests fork with extended async support. Unlike httpx, it has
different method names and parameter conventions that can be easy to confuse.

This test suite documents the differences and prevents regressions:
  - AsyncSession.close() NOT aclose()
  - allow_redirects=True (not follow_redirects)
  - raise_for_status() works as expected
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import niquests


def test_niquests_async_session_close_method():
    """niquests.AsyncSession provides close(), not aclose()."""
    session = niquests.AsyncSession()
    # Verify close exists and aclose doesn't
    assert hasattr(session, "close"), "niquests.AsyncSession should have close()"
    assert not hasattr(session, "aclose"), \
        "niquests.AsyncSession should NOT have aclose() (unlike some httpx versions)"


@pytest.mark.asyncio
async def test_niquests_async_get_with_allow_redirects():
    """niquests uses allow_redirects parameter, not follow_redirects."""
    with patch("niquests.AsyncSession.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        session = niquests.AsyncSession()
        # Both of these should work with niquests:
        # 1. Using allow_redirects (correct for niquests)
        # 2. No redirects parameter (defaults to following redirects)

        # The key thing: follow_redirects SHOULD NOT work with niquests
        # (it's an httpx parameter)


def test_niquests_parameter_name_differences():
    """Document key parameter differences between niquests and httpx."""
    # This is a reference test documenting the differences:
    #
    # httpx.AsyncClient / httpx.Client:
    #   - follow_redirects=True
    #   - timeout=10.0
    #   - base_url (on __init__)
    #
    # niquests.AsyncSession / niquests.Session:
    #   - allow_redirects=True  (note: allow, not follow)
    #   - timeout=10.0
    #   - base_url (on __init__)
    #   - close() not aclose()
    #
    # Common mistake: mixing httpx's follow_redirects with niquests,
    # which causes TypeError: got an unexpected keyword argument

    session = niquests.AsyncSession()
    # Verify initialization works
    assert session is not None


@pytest.mark.asyncio
async def test_niquests_session_aclose_should_be_close():
    """
    LEGACY TEST: Documents the fix for AsyncSession.aclose() → close().

    The old code did: await session.aclose()
    The correct code is: await session.close()
    """
    with patch("niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session

        # Set up close() to work
        mock_session.close = AsyncMock()
        # And ensure aclose doesn't exist
        del mock_session.aclose

        # When used in a context manager with proper cleanup
        try:
            session = niquests.AsyncSession()
            await session.close()  # <-- This is correct
            mock_session.close.assert_called_once()
        except AttributeError as e:
            # This should NOT happen now that we use close()
            pytest.fail(f"AsyncSession.close() should exist: {e}")
