"""Tag lookup errors reach the boundaries that own fallback behavior."""

from unittest.mock import AsyncMock

import niquests
import pytest

from paperless_common.paperless import PaperlessClient
from paperless_ai.search import webhook
from paperless_listener import app as listener


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [niquests.ConnectionError("offline"), niquests.HTTPError("503")]
)
async def test_tag_transport_failure_uses_listener_and_chat_fallback(
    monkeypatch, error
):
    """A real tag resolver failure uses payload tags and unavailable source cards."""
    async with PaperlessClient("http://unused", "token") as client:
        client.get_document = AsyncMock(return_value={"id": 42, "tags": [7]})
        client.get_document_for_chat = AsyncMock(return_value={"id": 42, "tags": [7]})
        client._get_all_tags = AsyncMock(side_effect=error)
        with pytest.raises(niquests.RequestException):
            await client.get_tag_names([7])
        monkeypatch.setattr(listener, "_paperless_client", client)
        tags = await listener._get_current_document_tags(42, {"ai:run-metadata"})
        assert tags == {"ai:run-metadata"}
        assert listener._route_to_stage(tags) == "paperless-ai:queue:metadata"

        monkeypatch.setattr(webhook, "_paperless_client", client)
        sources = await webhook._build_chat_sources({42: {"matched": True}})
        assert sources[0]["available"] is False
        restored = await webhook._restore_chat_sources(
            [{"id": 42, "title": "Saved title", "tag_names": ["Saved tag"]}]
        )
        assert restored[0]["available"] is False
        assert restored[0]["title"] == "Saved title"
        assert restored[0]["tag_names"] == ["Saved tag"]


@pytest.mark.asyncio
async def test_tag_lookup_does_not_hide_programming_errors(monkeypatch):
    """Programming errors propagate through the resolver and listener boundary."""
    async with PaperlessClient("http://unused", "token") as client:
        client.get_document = AsyncMock(return_value={"id": 42, "tags": [7]})
        client._get_all_tags = AsyncMock(side_effect=ValueError("bug"))
        monkeypatch.setattr(listener, "_paperless_client", client)
        with pytest.raises(ValueError, match="bug"):
            await listener._get_current_document_tags(42, {"ai:run-metadata"})


@pytest.mark.asyncio
async def test_listener_and_chat_refresh_metadata_on_each_operation(monkeypatch):
    """Long-lived clients see renamed tags and source metadata on the next request."""
    from unittest.mock import MagicMock

    async with PaperlessClient("http://unused", "token") as client:
        document = {"id": 42, "tags": [7], "correspondent": 7}
        client.get_document = AsyncMock(return_value=document)
        client.get_document_for_chat = AsyncMock(return_value=document)
        response = MagicMock()
        response.json.return_value = {"results": [{"id": 7, "name": "Old"}]}
        client._client.get = AsyncMock(return_value=response)
        monkeypatch.setattr(listener, "_paperless_client", client)
        monkeypatch.setattr(webhook, "_paperless_client", client)
        assert await listener._get_current_document_tags(42, set()) == {"Old"}
        sources = await webhook._build_chat_sources({42: {}, 43: {}})
        assert all(source["correspondent_name"] == "Old" for source in sources)
        assert client._client.get.await_count == 3  # listener tags + two chat lists

        response.json.return_value = {"results": [{"id": 7, "name": "New"}]}
        assert await listener._get_current_document_tags(42, set()) == {"New"}
        restored = await webhook._restore_chat_sources(sources)
        assert all(source["tag_names"] == ["New"] for source in restored)
        assert all(source["correspondent_name"] == "New" for source in restored)
        assert client._client.get.await_count == 6
