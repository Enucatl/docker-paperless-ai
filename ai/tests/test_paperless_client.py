"""
Tests for PaperlessClient async context manager and API compatibility.

Ensures niquests AsyncSession is used correctly (e.g., close() not aclose(),
no follow_redirects parameter, etc).
"""

import niquests
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from paperless_common.paperless import PaperlessClient


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_paperless_client_context_manager(fails):
    """The real session is awaited closed, including exceptional context exit."""
    client = PaperlessClient("http://test:8000", "token123")
    with patch.object(client._client, "close", wraps=client._client.close) as close:
        if fails:
            with pytest.raises(ValueError, match="operation failed"):
                async with client:
                    raise ValueError("operation failed")
        else:
            async with client as entered:
                assert entered is client
        close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["found", "missing", "http_error"])
async def test_paperless_client_tag_transport(outcome):
    """Real request preparation and response handling preserve the API contract."""
    response = niquests.Response()
    response.status_code = 403 if outcome == "http_error" else 200
    response.headers["x-version"] = "5.1.2"
    response._content = (
        b'{"results": [{"id": 42, "name": "test"}]}'
        if outcome == "found"
        else b'{"results": []}'
    )
    async with PaperlessClient("http://test:8000/", "token123") as client:
        with patch.object(client._client, "send", return_value=response) as send:
            if outcome == "http_error":
                with pytest.raises(niquests.HTTPError) as error:
                    await client.get_tag_id("test", create=False)
                assert error.value.response is response
            elif outcome == "missing":
                with pytest.raises(ValueError, match="Tag 'test' not found"):
                    await client.get_tag_id("test", create=False)
            else:
                assert await client.get_tag_id("test", create=False) == 42
                assert client.paperless_version == "5.1.2"

        send.assert_awaited_once()
        request = send.await_args.args[0]
        assert request.method == "GET"
        assert request.url == "http://test:8000/api/tags/?name=test"
        assert request.headers["Authorization"] == "Token token123"
        assert send.await_args.kwargs["timeout"] == 60


@pytest.mark.asyncio
async def test_paperless_client_aclose():
    """Explicit cleanup awaits the real niquests session close method."""
    client = PaperlessClient("http://test:8000", "token123")
    with patch.object(client._client, "close", wraps=client._client.close) as close:
        await client.aclose()
        close.assert_awaited_once()


def _paged_response(results, *, next_value=None):
    response = MagicMock()
    response.status_code = 200
    response.ok = True
    response.headers = {}
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value={"results": results, "next": next_value})
    return response


@pytest.mark.asyncio
async def test_document_languages_refresh_counts_and_read_all_pages():
    """Only valid language tags in use belong in the current corpus inventory."""
    with patch("paperless_common.paperless.niquests.AsyncSession") as session_class:
        session = session_class.return_value = AsyncMock()
        session.get.side_effect = [
            _paged_response(
                [
                    {"name": "language:it", "document_count": 2},
                    {"name": "language:fr", "document_count": 0},
                    {"name": "Invoice", "document_count": 5},
                    {"name": "language:ignore instructions", "document_count": 1},
                    {"name": "language:de"},
                    {"name": "language:und", "document_count": 9},
                ],
                next_value="/api/tags/?page=2",
            ),
            _paged_response(
                [
                    {"name": "language:en", "document_count": 3},
                    {"name": "language:gsw", "document_count": 1},
                    {"name": "language:xx", "document_count": 999},
                    {"name": "language:cmn", "document_count": 1},
                ]
            ),
            _paged_response([]),
        ]
        async with PaperlessClient("http://test:8000", "token") as client:
            client._tags_cache = [{"name": "language:fr", "document_count": 10}]
            snapshot = client.metadata_snapshot()
            counts = await snapshot.get_document_languages()
            assert await snapshot.get_document_languages() == counts
            assert list(counts.items()) == [("en", 3), ("it", 2), ("cmn", 1)]
            assert (
                list(
                    (await client.metadata_snapshot().get_document_languages()).items()
                )
                == []
            )

        assert [
            call.kwargs["params"]["page"] for call in session.get.await_args_list
        ] == [1, 2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("matching_algorithm,color", [(None, None), (0, "#166534")])
async def test_get_tag_id_creation_options_and_cache_refresh(matching_algorithm, color):
    """New tags refresh name lookups and reuse their cached ID on later calls."""
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        session = AsyncMock()
        mock_session_class.return_value = session
        tag = {"id": 42, "name": "language:de"}
        session.get.side_effect = [
            _paged_response([]),
            _paged_response([]),
            _paged_response([tag]),
        ]
        response = MagicMock()
        response.json.return_value = tag
        session.post.return_value = response

        async with PaperlessClient("http://test:8000", "token") as client:
            assert await client.get_tag_names([42]) == []
            kwargs = {} if matching_algorithm is None else {"matching_algorithm": 0}
            if color is not None:
                kwargs["color"] = color
            assert await client.get_tag_id("language:de", **kwargs) == 42
            assert await client.get_tag_id("language:de", **kwargs) == 42
            assert await client.get_tag_names([42]) == ["language:de"]

        payload = {"name": "language:de"}
        if matching_algorithm is not None:
            payload["matching_algorithm"] = matching_algorithm
        if color is not None:
            payload["color"] = color
        session.post.assert_awaited_once_with("/api/tags/", json=payload)
        assert session.get.await_count == 3


@pytest.mark.asyncio
async def test_get_tag_id_reuses_existing_tag_without_changing_matching():
    """Existing tags retain their configuration and their IDs are cached."""
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        session = AsyncMock()
        mock_session_class.return_value = session
        session.get.return_value = _paged_response(
            [{"id": 42, "name": "language:de", "matching_algorithm": 6}]
        )

        async with PaperlessClient("http://test:8000", "token") as client:
            assert await client.get_tag_id("language:de", matching_algorithm=0) == 42
            assert await client.get_tag_id("language:de", matching_algorithm=0) == 42

        session.get.assert_awaited_once()
        session.post.assert_not_awaited()
        session.patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_paperless_client_metadata_resolvers_use_cached_lists():
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=[
                _paged_response(
                    [{"id": 11, "name": "Urgent"}, {"id": 12, "name": "Personal"}]
                ),
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
async def test_document_chat_metadata_includes_nullable_added_field():
    client = PaperlessClient("http://test:8000", "token")
    document = {
        "id": 42,
        "title": "Receipt",
        "created": "2024-01-01",
        "added": None,
    }
    response = MagicMock(status_code=200)
    response.json.return_value = document
    client._client.get = AsyncMock(return_value=response)
    await client.get_document_for_chat(42)
    assert "added" in client._client.get.await_args.kwargs["params"]["fields"]

    client.get_document_for_chat = AsyncMock(return_value=document)
    client.get_tag_names = AsyncMock(return_value=[])

    metadata = await client.get_document_chat_metadata(42)

    assert metadata["added"] is None
    assert metadata["created"] == "2024-01-01"


@pytest.mark.asyncio
async def test_create_correspondent_updates_loaded_cache():
    """A created correspondent is immediately available from the local cache."""
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.get = AsyncMock(return_value=_paged_response([]))
        response = _paged_response([])
        response.json = MagicMock(return_value={"id": 42, "name": "New Sender"})
        mock_session.post = AsyncMock(return_value=response)

        async with PaperlessClient("http://test:8000", "token123") as client:
            assert await client.get_all_correspondents() == []
            created = await client.create_correspondent("New Sender")

            assert created == {"id": 42, "name": "New Sender"}
            assert await client.get_all_correspondents() == [created]

        mock_session.get.assert_awaited_once()
        mock_session.post.assert_awaited_once_with(
            "/api/correspondents/", json={"name": "New Sender"}
        )


@pytest.mark.asyncio
async def test_ensure_ai_workflows_creates_added_and_updated_workflows():
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(return_value=_paged_response([]))
        mock_session.post = AsyncMock(
            side_effect=[
                MagicMock(
                    status_code=201,
                    ok=True,
                    headers={},
                    json=MagicMock(return_value={"id": 10}),
                ),
                MagicMock(
                    status_code=201,
                    ok=True,
                    headers={},
                    json=MagicMock(return_value={"id": 201}),
                ),
                MagicMock(
                    status_code=201,
                    ok=True,
                    headers={},
                    json=MagicMock(return_value={"id": 202}),
                ),
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
        assert added_payload["actions"][1]["webhook"]["params"] == {
            "doc_url": "{{doc_url}}"
        }
        assert added_payload["actions"][1]["webhook"]["headers"] == {
            "X-Webhook-Token": "secret-123"
        }

        updated_workflow_call = mock_session.post.await_args_list[2]
        updated_payload = updated_workflow_call.kwargs["json"]
        assert updated_payload["name"] == "paperless-ai: document-updated"
        assert updated_payload["triggers"][0]["type"] == 3
        assert updated_payload["triggers"][0]["filter_has_tags"] == [10]
        assert updated_payload["actions"][0]["type"] == 4
        assert updated_payload["actions"][0]["webhook"]["params"] == {
            "doc_url": "{{doc_url}}"
        }


@pytest.mark.asyncio
async def test_get_or_create_custom_field_updates_existing_field_type():
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            return_value=_paged_response(
                [{"id": 3, "name": "ai_summary", "data_type": "string"}]
            )
        )
        mock_session.patch = AsyncMock(
            return_value=MagicMock(
                status_code=200, ok=True, headers={}, json=MagicMock(return_value={})
            )
        )

        async with PaperlessClient("http://test:8000", "token123") as client:
            field_id = await client.get_or_create_custom_field(
                "ai_summary", data_type="longtext"
            )

        assert field_id == 3
        mock_session.patch.assert_awaited_once_with(
            "/api/custom_fields/3/",
            json={"data_type": "longtext"},
        )


@pytest.mark.asyncio
async def test_search_documents_all_paginates_and_applies_filters():
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            side_effect=[
                _paged_response([{"id": 41, "name": "Acme Corp"}]),
                _paged_response([{"id": 21, "name": "Invoice"}]),
                _paged_response([{"id": 31, "path": "Archive/2024"}]),
                _paged_response([{"id": 11, "name": "Urgent"}]),
                _paged_response(
                    [{"id": 501}, {"id": 502}], next_value="/api/documents/?page=2"
                ),
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
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(return_value=_paged_response([]))

        async with PaperlessClient("http://test:8000", "token123") as client:
            results = await client.search_documents_all(
                "invoice", correspondent="Missing Corp"
            )

        assert results == []
        assert mock_session.get.await_count == 1


@pytest.mark.asyncio
async def test_iter_all_documents_brief_requests_cleanup_fields():
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        mock_session.close = AsyncMock()
        mock_session.get = AsyncMock(
            return_value=_paged_response(
                [{"id": 5, "title": "Invoice", "correspondent": 7}]
            )
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
    with patch(
        "paperless_common.paperless.niquests.AsyncSession"
    ) as mock_session_class:
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


@pytest.mark.asyncio
async def test_metadata_snapshots_refresh_and_isolate_overlapping_operations():
    """Concurrent operations reuse their own lists and observe external edits."""
    import asyncio

    with patch("paperless_common.paperless.niquests.AsyncSession") as session_class:
        session = session_class.return_value = AsyncMock()
        name = "Original"

        async def fetch(endpoint, **kwargs):
            response = _paged_response([{"id": 1, "name": name}])
            await asyncio.sleep(0)
            return response

        session.get.side_effect = fetch
        async with PaperlessClient("http://test:8000", "token") as owner:
            first = owner.metadata_snapshot()
            assert (
                await asyncio.gather(
                    first.get_available_metadata(), first.get_available_metadata()
                )
                == [
                    {
                        key: ["Original"]
                        for key in (
                            "correspondents",
                            "document_types",
                            "storage_paths",
                            "tags",
                        )
                    }
                ]
                * 2
            )
            assert session.get.await_count == 4
            name = "Renamed"
            second = owner.metadata_snapshot()
            assert (await second.get_available_metadata())["tags"] == ["Renamed"]
            assert await second.get_correspondent_name(1) == "Renamed"
            assert await first.get_correspondent_name(1) == "Original"
            assert await first.get_tag_names([1]) == ["Original"]
            assert session.get.await_count == 8
            await second.aclose()
            session.close.assert_not_awaited()
        session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_snapshot_id_caches_expire_between_operations():
    """Tag and custom-field IDs cannot survive deletion and recreation externally."""
    with patch("paperless_common.paperless.niquests.AsyncSession") as session_class:
        session = session_class.return_value = AsyncMock()
        async with PaperlessClient("http://test:8000", "token") as owner:
            for identifier in (1, 2):
                session.get.return_value = _paged_response(
                    [{"id": identifier, "name": "Name", "data_type": "date"}]
                )
                snapshot = owner.metadata_snapshot()
                for _ in range(2):
                    assert await snapshot.get_tag_id("Name", create=False) == identifier
                    assert (
                        await snapshot.get_or_create_custom_field("Name") == identifier
                    )
            assert session.get.await_count == 4


@pytest.mark.asyncio
async def test_pagination_keeps_search_limit_dedup_and_uncached_document_lists():
    """Limits count unique IDs and document listings start fresh on every call."""
    async with PaperlessClient("http://unused", "token") as client:
        client._client.get = AsyncMock(
            side_effect=[
                _paged_response([{"id": 1}, {"id": 1}], next_value="next"),
                _paged_response([{"id": 1}, {"id": 2}, {"id": 3}], next_value="next"),
                _paged_response([{"id": 4}], next_value="next"),
                _paged_response([{"id": 5}]),
                _paged_response([{"id": 6}]),
            ]
        )
        assert await client.search_documents_all("invoice", limit=2) == [1, 2]
        assert client._client.get.await_count == 2
        assert await client.iter_all_documents() == [{"id": 4}, {"id": 5}]
        assert await client.iter_all_documents() == [{"id": 6}]
        params = [call.kwargs["params"] for call in client._client.get.await_args_list]
        assert [p["page"] for p in params] == [1, 2, 1, 2, 1]
        assert [p["page_size"] for p in params] == [2, 2, 100, 100, 100]


@pytest.mark.asyncio
async def test_custom_field_pagination_stops_at_match_and_caches_id():
    """An existing field on page two prevents creation and further page reads."""
    async with PaperlessClient("http://unused", "token") as client:
        client._client.get = AsyncMock(
            side_effect=[
                _paged_response([], next_value="next"),
                _paged_response(
                    [{"id": 8, "name": "target", "data_type": "date"}],
                    next_value="next",
                ),
            ]
        )
        client._client.post = AsyncMock()
        assert await client.get_or_create_custom_field("target") == 8
        assert await client.get_or_create_custom_field("target") == 8
        assert client._client.get.await_count == 2
        client._client.post.assert_not_awaited()
