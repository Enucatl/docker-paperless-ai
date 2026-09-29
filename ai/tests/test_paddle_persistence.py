"""Worker ownership, concurrency, and real Paddle JSON persistence checks."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from paperless_ai.agents.smart_graph_agent import _ExtractedMetadata
from paperless_ai.core import runner
from paperless_ai.core.config import AgentConfig
from paperless_common.queue import TaskQueues
from tests.conftest import PAPERLESS_URL, _make_test_pdf, _upload_document

_METADATA = {
    "pipeline": "PaddleOCR-VL-1.6",
    "model": "PaddlePaddle/PaddleOCR-VL-1.6",
    "layout_model": "PP-DocLayoutV3",
}


@pytest.fixture(autouse=True)
def mock_ocr_metadata(monkeypatch):
    """Provide a fresh deployment identity for mocked OCR batches."""
    fetch = AsyncMock(return_value=_METADATA)
    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.fetch_ocr_metadata", fetch)
    return fetch


def _config(**kwargs) -> AgentConfig:
    """Build a configuration independent of installed inference endpoints."""
    return AgentConfig(
        **{
            "ocr_endpoint": "http://paddle:8080",
            "metadata_endpoint": None,
            "metadata_model": "test-metadata",
            "chat_model": "test-chat",
            **kwargs,
        }
    )


def _queues(doc_ids: set[int] | None = None) -> AsyncMock:
    """Provide the queue operations used by the batch workers."""
    queues = AsyncMock()
    queues.stage_size.return_value = len(doc_ids or {42})
    queues.peek_stage.return_value = doc_ids or {42}
    queues.mark_failure.return_value = (1, False)
    return queues


def _client() -> MagicMock:
    """Provide old and freshly fetched document state with distinct fields."""
    client = MagicMock()
    client.metadata_snapshot.return_value = client
    initial = {
        "id": 42,
        "content": "Original transcript",
        "tags": [1, 3],
        "custom_fields": [
            {"field": 10, "value": '{"previous":"paddle"}'},
            {"field": 11, "value": '{"previous":"metadata"}'},
        ],
    }
    refreshed = {
        **initial,
        "tags": [1, 3, 4],
        "custom_fields": initial["custom_fields"]
        + [{"field": 12, "value": "Added during processing"}],
    }
    client.get_document = AsyncMock(side_effect=[initial, refreshed])
    client.get_document_with_content = AsyncMock(return_value=initial)
    client.download_original = AsyncMock(return_value=b"mock PDF")
    client.get_tag_id = AsyncMock(
        side_effect=lambda name, **kwargs: {
            "ai:run-ocr": 1,
            "ai:run-metadata": 2,
            "language:en": 5,
        }[name]
    )
    client.get_or_create_custom_field = AsyncMock(return_value=10)
    client.patch_document = AsyncMock()
    client._get_all_tags = AsyncMock(return_value=[])
    client.paperless_version = "test"
    return client


async def test_ocr_atomic_patch_owns_only_content_and_ocr_output(
    monkeypatch, mock_ocr_metadata
) -> None:
    """Document OCR preserves fresh metadata, fields, and unrelated tags."""
    client, queues = _client(), _queues()
    output = {
        "schema_version": 1,
        **_METADATA,
        "pages": [{"page_index": 0, "markdown": "新"}],
    }
    monkeypatch.setattr(runner, "_check_server_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "paperless_ai.agents.paddle_ocr.run_paddle_ocr",
        AsyncMock(return_value=("New transcript", output, 1, 0.1)),
    )

    assert await runner.run_ocr_batch(client, _config(), queues) == (
        1,
        0,
    )

    client.get_or_create_custom_field.assert_awaited_once_with(
        "ai_ocr_output", data_type="longtext"
    )
    client.patch_document.assert_awaited_once()
    doc_id, payload = client.patch_document.await_args.args
    assert doc_id == 42
    assert set(payload) == {"content", "tags", "custom_fields"}
    assert payload["content"] == "New transcript"
    assert set(payload["tags"]) == {2, 3, 4}
    fields = {field["field"]: field["value"] for field in payload["custom_fields"]}
    assert fields[11] == '{"previous":"metadata"}'
    assert fields[12] == "Added during processing"
    assert json.loads(fields[10]) == output
    queues.remove.assert_awaited_once_with(42, TaskQueues.KEY_OCR)
    queues.enqueue_metadata.assert_awaited_once_with(42)
    mock_ocr_metadata.assert_awaited_once()


async def test_ocr_batch_fetches_metadata_once(monkeypatch, mock_ocr_metadata) -> None:
    """All documents in one batch share the same fetched identity."""
    client, queues = _client(), _queues({42, 43})
    client.get_document.side_effect = None
    client.get_document.return_value = {"id": 42}
    parser = AsyncMock(return_value=("Text", {**_METADATA}, 1, 0.1))
    monkeypatch.setattr(runner, "_check_server_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.run_paddle_ocr", parser)

    assert await runner.run_ocr_batch(client, _config(dry_run=True), queues) == (2, 0)
    mock_ocr_metadata.assert_awaited_once()
    assert parser.await_count == 2
    assert all(call.args[2] is _METADATA for call in parser.await_args_list)
    client.patch_document.assert_not_awaited()

    assert await runner.run_ocr_batch(client, _config(dry_run=True), queues) == (2, 0)
    assert mock_ocr_metadata.await_count == 2


async def test_bad_batch_metadata_preserves_documents(
    monkeypatch, mock_ocr_metadata
) -> None:
    """Metadata failure aborts the batch before downloads or document writes."""
    client, queues = _client(), _queues()
    monkeypatch.setattr(runner, "_check_server_reachable", AsyncMock(return_value=True))
    mock_ocr_metadata.side_effect = ValueError("OCR /metadata requires nonempty model")

    with pytest.raises(ValueError, match="OCR /metadata requires nonempty model"):
        await runner.run_ocr_batch(client, _config(), queues)
    client.get_document.assert_not_awaited()
    client.download_original.assert_not_awaited()
    client.patch_document.assert_not_awaited()
    queues.remove.assert_not_awaited()


@pytest.mark.parametrize(
    "failure", [ValueError("missing pages"), TimeoutError("timeout")]
)
async def test_paddle_failure_preserves_document_and_schedules_retry(
    monkeypatch, failure: Exception
) -> None:
    """An incomplete response or timeout never initiates a document PATCH."""
    client, queues = _client(), _queues()
    monkeypatch.setattr(runner, "_check_server_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "paperless_ai.agents.paddle_ocr.run_paddle_ocr", AsyncMock(side_effect=failure)
    )

    assert await runner.run_ocr_batch(client, _config(), queues) == (0, 1)

    client.patch_document.assert_not_awaited()
    queues.remove.assert_not_awaited()
    queues.enqueue_metadata.assert_not_awaited()
    queues.mark_failure.assert_awaited_once_with(42, TaskQueues.KEY_OCR, max_attempts=3)


@pytest.mark.parametrize(
    "status,ready", [(200, True), (204, True), (404, False), (500, False)]
)
async def test_paddle_readiness_requires_successful_health(
    monkeypatch, status: int, ready: bool
) -> None:
    """A missing health route is not evidence that the Paddle pipeline is ready."""
    session = MagicMock()
    session.get = AsyncMock(return_value=MagicMock(status_code=status))
    monkeypatch.setattr(runner, "_get_model_probe_session", lambda: session)
    monkeypatch.setattr(runner, "_offline_servers", set())

    assert (
        await runner._check_server_reachable("http://paddle:8080/", strict_health=True)
        is ready
    )

    session.get.assert_awaited_once_with("http://paddle:8080/health")


async def test_metadata_preserves_fresh_ocr_output(monkeypatch) -> None:
    """Metadata reads content while preserving freshly fetched OCR-owned fields."""
    client, queues = _client(), _queues()
    client.get_document.side_effect = None
    client.get_document.return_value = {
        "id": 42,
        "tags": [2, 3, 4],
        "custom_fields": [
            {"field": 10, "value": '{"fresh":"Paddle output"}'},
            {"field": 12, "value": "Unrelated field"},
        ],
    }
    strategy = MagicMock()
    strategy.extract = AsyncMock(
        return_value=_ExtractedMetadata(
            title="New title", summary="New summary", languages=["en"]
        )
    )
    monkeypatch.setattr(
        "paperless_ai.agents.smart_graph_agent.StructuredOutputStrategy",
        lambda: strategy,
    )

    assert await runner.run_metadata_batch(client, _config(), queues, 13, 14, 11) == (
        1,
        0,
    )

    assert "Original transcript" in strategy.extract.await_args.args[0]
    payload = client.patch_document.await_args.args[1]
    assert "content" not in payload
    fields = {field["field"]: field["value"] for field in payload["custom_fields"]}
    assert fields[10] == '{"fresh":"Paddle output"}'
    assert fields[12] == "Unrelated field"
    assert json.loads(fields[11])["ai_metadata"]["title"] == "New title"
    assert set(payload["tags"]) == {3, 4, 5}


async def test_stages_serialize_same_document_and_allow_other_documents() -> None:
    """Metadata waits for the same document's OCR without blocking another ID."""
    ocr_entered, release_ocr, other_entered = (asyncio.Event() for _ in range(3))
    same_document_entered = False

    async def ocr(doc_id: int) -> bool:
        """Hold one document until the test permits OCR to finish."""
        ocr_entered.set()
        await release_ocr.wait()
        return True

    async def metadata(doc_id: int) -> bool:
        """Record which metadata jobs can enter while OCR is in progress."""
        nonlocal same_document_entered
        if doc_id == 42:
            same_document_entered = True
        else:
            other_entered.set()
        return True

    tasks = [
        asyncio.create_task(
            runner._run_stage("OCR", None, TaskQueues.KEY_OCR, _queues({42}), ocr)
        )
    ]
    try:
        await asyncio.wait_for(ocr_entered.wait(), timeout=1)
        tasks.append(
            asyncio.create_task(
                runner._run_stage(
                    "Metadata",
                    None,
                    TaskQueues.KEY_METADATA,
                    _queues({42, 43}),
                    metadata,
                )
            )
        )
        await asyncio.wait_for(other_entered.wait(), timeout=1)
        assert not same_document_entered
        release_ocr.set()
        assert await asyncio.wait_for(asyncio.gather(*tasks), timeout=1) == [
            (1, 0),
            (2, 0),
        ]
        assert same_document_entered
    finally:
        release_ocr.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.requires_redis
async def test_paddle_large_json_roundtrip_metadata_preservation_and_failed_rerun(
    paperless_client, task_queues, monkeypatch
) -> None:
    """Paperless stores large Unicode JSON and failures cannot partially replace it."""
    client = paperless_client
    token = client._client.headers["Authorization"].split(" ")[1]
    config = _config(paperless_url=PAPERLESS_URL, paperless_token=token)
    monkeypatch.setattr(runner, "_check_server_reachable", AsyncMock(return_value=True))
    text = "# Rechnung 发票\n\n| Item | Total |\n| --- | --- |\n| Café | €10 |"
    output = {
        "schema_version": 1,
        "model": "PaddleOCR-VL-1.6",
        "pages": [
            {
                "page_index": 0,
                "markdown": text,
                "prunedResult": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "日本語 Καλημέρα résumé\n" * 12000,
                        }
                    ]
                },
            }
        ],
    }
    assert len(json.dumps(output, ensure_ascii=False).encode()) > 100_000
    paddle = AsyncMock(return_value=(text, output, 1, 0.1))
    monkeypatch.setattr("paperless_ai.agents.paddle_ocr.run_paddle_ocr", paddle)
    fields = {}
    for name, data_type in (
        ("ai_ocr_output", "longtext"),
        ("ai_result", "longtext"),
        ("ai_summary", "longtext"),
        ("ai_processed", "date"),
        ("paddle-test-preserved", "longtext"),
    ):
        fields[name] = await client.get_or_create_custom_field(
            name, data_type=data_type
        )
    tag_ocr = await client.get_tag_id(config.tag_ocr)
    tag_metadata = await client.get_tag_id(config.tag_metadata)
    unrelated_tag = await client.get_tag_id("integration:paddle-preserved")
    doc_id = await _upload_document(client, _make_test_pdf())
    try:
        await client.patch_document(
            doc_id,
            {
                "title": "Keep existing title",
                "content": "Old content",
                "tags": [tag_ocr, unrelated_tag],
                "custom_fields": [
                    {"field": fields["ai_result"], "value": '{"existing":"metadata"}'},
                    {"field": fields["paddle-test-preserved"], "value": "Preserved ✓"},
                ],
            },
        )
        await task_queues.enqueue_ocr(doc_id)
        assert await runner.run_ocr_batch(client, config, task_queues) == (1, 0)
        document = await client.get_document_with_content(doc_id)
        values = {item["field"]: item["value"] for item in document["custom_fields"]}
        assert document["content"] == text
        assert document["title"] == "Keep existing title"
        assert json.loads(values[fields["ai_ocr_output"]]) == output
        assert values[fields["ai_result"]] == '{"existing":"metadata"}'
        assert values[fields["paddle-test-preserved"]] == "Preserved ✓"
        assert set(document["tags"]) == {tag_metadata, unrelated_tag}
        response = await client._client.get(
            f"/api/custom_fields/{fields['ai_ocr_output']}/"
        )
        response.raise_for_status()
        assert response.json()["data_type"] == "longtext"

        strategy = MagicMock()
        strategy.extract = AsyncMock(
            return_value=_ExtractedMetadata(
                title="New metadata title", summary="Résumé", languages=["en"]
            )
        )
        monkeypatch.setattr(
            "paperless_ai.agents.smart_graph_agent.StructuredOutputStrategy",
            lambda: strategy,
        )
        assert await runner.run_metadata_batch(
            client,
            config,
            task_queues,
            fields["ai_processed"],
            fields["ai_summary"],
            fields["ai_result"],
        ) == (1, 0)
        document = await client.get_document_with_content(doc_id)
        values = {item["field"]: item["value"] for item in document["custom_fields"]}
        assert document["content"] == text
        assert document["title"] == "New metadata title"
        assert json.loads(values[fields["ai_ocr_output"]]) == output
        assert values[fields["paddle-test-preserved"]] == "Preserved ✓"
        assert unrelated_tag in document["tags"]
        assert tag_metadata not in document["tags"]

        await client.patch_document(doc_id, {"tags": [*document["tags"], tag_ocr]})
        before = await client.get_document_with_content(doc_id)
        paddle.side_effect = ValueError("incomplete Paddle page coverage")
        await task_queues.enqueue_ocr(doc_id)
        assert await runner.run_ocr_batch(client, config, task_queues) == (0, 1)
        after = await client.get_document_with_content(doc_id)
        for key in (
            "content",
            "title",
            "custom_fields",
            "tags",
            "correspondent",
            "created",
        ):
            assert after.get(key) == before.get(key)
        assert doc_id not in await task_queues.peek_stage(TaskQueues.KEY_OCR)
        assert await task_queues.release_due(TaskQueues.KEY_OCR, now=9999999999) == 1
        assert doc_id in await task_queues.peek_stage(TaskQueues.KEY_OCR)
    finally:
        await client._client.delete(f"/api/documents/{doc_id}/")
