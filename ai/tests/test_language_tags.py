"""Metadata language tag persistence without external services."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from paperless_ai.agents.smart_graph_agent import _ExtractedMetadata
from paperless_ai.core.config import AgentConfig
from paperless_ai.core.runner import run_metadata_batch
from paperless_common.paperless import PaperlessClient
from paperless_common.queue import TaskQueues


@pytest.mark.parametrize("mode", ["known", "unknown", "failure", "dry_run"])
async def test_language_tags_are_atomic_and_reused_across_documents(mode: str) -> None:
    """Concurrent metadata writes preserve tags and retry incomplete resolution."""
    config = AgentConfig(
        metadata_model="metadata",
        chat_model="chat",
        dry_run=mode == "dry_run",
        ocr_concurrency=2,
    )
    queues = AsyncMock(spec=TaskQueues)
    queues.stage_size.return_value = 2
    queues.peek_stage.return_value = {1, 2}
    queues.mark_failure.return_value = (1, False)
    extracted = _ExtractedMetadata(
        title="New title", languages=None if mode == "unknown" else ["de", "en"]
    )
    strategy = MagicMock(extract=AsyncMock(return_value=extracted))
    session = AsyncMock()
    response = MagicMock()
    response.headers = {}
    response.json.return_value = {"results": []}
    session.get.return_value = response

    async def create_tag(endpoint: str, *, json: dict) -> MagicMock:
        """Yield during creation so an unlocked cache lookup would race."""
        await asyncio.sleep(0)
        assert endpoint == "/api/tags/"
        assert json["matching_algorithm"] == 0
        assert json["color"] == "#166534"
        if mode == "failure" and json["name"] == "language:de":
            raise RuntimeError("tag service unavailable")
        created = MagicMock()
        created.json.return_value = {
            "id": {"language:de": 20, "language:und": 22}[json["name"]]
        }
        return created

    session.post.side_effect = create_tag
    with (
        patch("paperless_common.paperless.niquests.AsyncSession", return_value=session),
        patch(
            "paperless_ai.agents.smart_graph_agent.StructuredOutputStrategy",
            return_value=strategy,
        ),
    ):
        async with PaperlessClient("http://paperless", "token") as client:
            client._tag_id_cache[config.tag_metadata] = 10
            client.get_document_with_content = AsyncMock(
                return_value={
                    "content": "Rechnung. Please pay the invoice.",
                    "tags": [10, 11, 12],
                    "custom_fields": [{"field": 99, "value": "keep"}],
                }
            )
            client.get_document = AsyncMock(
                return_value=client.get_document_with_content.return_value
            )
            client._get_all_tags = AsyncMock(
                return_value=[
                    {"id": 11, "name": "language:fr"},
                    {"id": 12, "name": "keep"},
                ]
            )
            client.patch_document = AsyncMock()
            result = await run_metadata_batch(client, config, queues, 1, 2, 3)

            assert result == ((0, 2) if mode == "failure" else (2, 0))
            if mode in {"failure", "dry_run"}:
                client.patch_document.assert_not_awaited()
                queues.remove.assert_not_awaited()
            else:
                assert client.patch_document.await_count == 2
                for call in client.patch_document.await_args_list:
                    payload = call.args[1]
                    assert payload["tags"] == [12, 22 if mode == "unknown" else 20]
                    assert "content" not in payload
                    fields = {
                        cf["field"]: cf["value"] for cf in payload["custom_fields"]
                    }
                    assert fields[99] == "keep"
                    assert (
                        json.loads(fields[3])["ai_metadata"]["languages"]
                        == extracted.languages
                    )
                assert queues.remove.await_count == 2
            if mode == "dry_run":
                session.post.assert_not_awaited()
                client._get_all_tags.assert_not_awaited()
            elif mode in {"known", "unknown"}:
                assert session.post.await_count == 1
                client._get_all_tags.assert_awaited_once_with(force=True)
            else:
                assert queues.mark_failure.await_count == 2
