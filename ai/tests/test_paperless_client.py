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


def _paged_response(results, *, next_value=None):
    response = MagicMock()
    response.status_code = 200
    response.ok = True
    response.headers = {}
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"results": results, "next": next_value})
    return response


@pytest.mark.asyncio
async def test_paperless_client_metadata_resolvers_use_cached_lists():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=[
                _paged_response([{"id": 11, "name": "Urgent"}, {"id": 12, "name": "Personal"}]),
                _paged_response([{"id": 21, "name": "Receipt"}]),
                _paged_response([{"id": 31, "path": "Archive/2023"}]),
                _paged_response([{"id": 41, "name": "Acme Corp"}]),
            ]
        )

        async with PaperlessClient("http://test:8000", "token123") as client:
            assert await client.get_tag_names([12, 11]) == ["Personal", "Urgent"]
            assert await client.get_document_type_name(21) == "Receipt"
            assert await client.get_storage_path_name(31) == "Archive/2023"

            metadata = await client.get_available_metadata()
            assert metadata == {
                "correspondents": ["Acme Corp"],
                "document_types": ["Receipt"],
                "storage_paths": ["Archive/2023"],
                "tags": ["Personal", "Urgent"],
            }

            assert await client.get_tag_names([11]) == ["Urgent"]
            assert await client.get_document_type_name(21) == "Receipt"
            assert await client.get_storage_path_name(31) == "Archive/2023"

        assert mock_session.get.await_count == 4


@pytest.mark.asyncio
async def test_ensure_ai_workflows_creates_added_and_updated_workflows():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(return_value=_paged_response([]))
        mock_session.post = AsyncMock(
            side_effect=[
                MagicMock(status_code=201, ok=True, headers={}, json=MagicMock(return_value={"id": 10})),
                MagicMock(status_code=201, ok=True, headers={}, json=MagicMock(return_value={"id": 201})),
                MagicMock(status_code=201, ok=True, headers={}, json=MagicMock(return_value={"id": 202})),
            ]
        )
        mock_session.patch = AsyncMock()

        async with PaperlessClient("http://test:8000", "token123") as client:
            added_id, updated_id = await client.ensure_ai_workflows(
                tag_ocr="ai:run-ocr",
                webhook_url="http://webhook-listener:8001/webhook/document",
                webhook_secret="secret-123",
            )

        assert (added_id, updated_id) == (201, 202)

        tag_create_call = mock_session.post.await_args_list[0]
        assert tag_create_call.args[0] == "/api/tags/"
        assert tag_create_call.kwargs["json"] == {"name": "ai:run-ocr"}

        added_workflow_call = mock_session.post.await_args_list[1]
        added_payload = added_workflow_call.kwargs["json"]
        assert added_payload["name"] == "paperless-ai: document-added"
        assert added_payload["actions"][0]["type"] == 1
        assert added_payload["actions"][0]["assign_tags"] == [10]
        assert added_payload["actions"][1]["type"] == 4
        assert added_payload["actions"][1]["webhook"]["params"] == {"doc_url": "{{doc_url}}"}
        assert added_payload["actions"][1]["webhook"]["headers"] == {"X-Webhook-Token": "secret-123"}

        updated_workflow_call = mock_session.post.await_args_list[2]
        updated_payload = updated_workflow_call.kwargs["json"]
        assert updated_payload["name"] == "paperless-ai: document-updated"
        assert updated_payload["triggers"][0]["type"] == 3
        assert updated_payload["triggers"][0]["filter_has_tags"] == [10]
        assert updated_payload["actions"][0]["type"] == 4
        assert updated_payload["actions"][0]["webhook"]["params"] == {"doc_url": "{{doc_url}}"}


@pytest.mark.asyncio
async def test_get_or_create_custom_field_updates_existing_field_type():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            return_value=_paged_response([{"id": 3, "name": "ai_summary", "data_type": "string"}])
        )
        mock_session.patch = AsyncMock(
            return_value=MagicMock(status_code=200, ok=True, headers={}, json=MagicMock(return_value={}))
        )

        async with PaperlessClient("http://test:8000", "token123") as client:
            field_id = await client.get_or_create_custom_field("ai_summary", data_type="longtext")

        assert field_id == 3
        mock_session.patch.assert_awaited_once_with(
            "/api/custom_fields/3/",
            json={"data_type": "longtext"},
        )


@pytest.mark.asyncio
async def test_search_documents_all_paginates_and_applies_filters():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=[
                _paged_response([{"id": 41, "name": "Acme Corp"}]),
                _paged_response([{"id": 21, "name": "Invoice"}]),
                _paged_response([{"id": 31, "path": "Archive/2024"}]),
                _paged_response([{"id": 11, "name": "Urgent"}]),
                _paged_response([{"id": 501}, {"id": 502}], next_value="/api/documents/?page=2"),
                _paged_response([{"id": 503}]),
            ]
        )

        async with PaperlessClient("http://test:8000", "token123") as client:
            results = await client.search_documents_all(
                "invoice",
                correspondent="Acme Corp",
                document_type="Invoice",
                storage_path="Archive/2024",
                tags=["Urgent"],
                year="2024",
            )

        assert results == [501, 502, 503]
        search_call = mock_session.get.await_args_list[4]
        assert search_call.args[0] == "/api/documents/"
        assert search_call.kwargs["params"] == {
            "query": "invoice",
            "fields": "id",
            "page_size": 250,
            "page": 1,
            "correspondent__id": 41,
            "document_type__id": 21,
            "storage_path__id": 31,
            "tags__id__in": "11",
            "created__year": "2024",
        }


@pytest.mark.asyncio
async def test_search_documents_all_returns_empty_when_filter_name_unknown():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(return_value=_paged_response([]))

        async with PaperlessClient("http://test:8000", "token123") as client:
            results = await client.search_documents_all("invoice", correspondent="Missing Corp")

        assert results == []
        assert mock_session.get.await_count == 1


@pytest.mark.asyncio
async def test_iter_all_documents_brief_requests_cleanup_fields():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            return_value=_paged_response([{"id": 5, "title": "Invoice", "correspondent": 7}])
        )

        async with PaperlessClient("http://test:8000", "token123") as client:
            results = await client.iter_all_documents_brief()

        assert results == [{"id": 5, "title": "Invoice", "correspondent": 7}]
        mock_session.get.assert_awaited_once_with(
            "/api/documents/",
            params={"page": 1, "page_size": 250, "fields": "id,title,correspondent"},
        )


@pytest.mark.asyncio
async def test_count_documents_for_correspondent_uses_count_field():
    with patch("paperless_ai.core.paperless.niquests.AsyncSession") as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        response = MagicMock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status = MagicMock()
        response.json = MagicMock(return_value={"count": 4, "results": []})
        mock_session.get = AsyncMock(return_value=response)

        async with PaperlessClient("http://test:8000", "token123") as client:
            count = await client.count_documents_for_correspondent(41)

        assert count == 4
        mock_session.get.assert_awaited_once_with(
            "/api/documents/",
            params={"correspondent__id": 41, "page_size": 1, "fields": "id"},
        )
