"""Fast regression checks for Paperless task API polling."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests import conftest


@pytest.mark.parametrize(
    "task",
    [
        {"status": "success", "result_data": {"document_id": 42}},
        {"status": "SUCCESS", "related_document": 42},
        {"status": "failure", "result_data": {"error_message": "consumer failed"}},
        {"status": "revoked"},
        {"status": "success", "result_data": {}},
        {"status": "started"},
    ],
)
async def test_upload_follows_its_task_and_fails_promptly(monkeypatch, task):
    """Accept both API versions and never poll a terminal task for two minutes."""
    post_response = MagicMock(text='"upload-task"')
    pending = MagicMock()
    pending.json.return_value = {"results": []}
    completed = MagicMock()
    matching = {"task_id": "upload-task", **task}
    tasks = [{"task_id": "unrelated", "status": "success"}, matching]
    completed.json.return_value = (
        tasks if task["status"] == "SUCCESS" else {"results": tasks}
    )
    session = SimpleNamespace(
        post=AsyncMock(return_value=post_response),
        get=AsyncMock(side_effect=[pending, completed]),
    )
    sleep = AsyncMock()
    monkeypatch.setattr(conftest, "asyncio", SimpleNamespace(sleep=sleep))
    monotonic = MagicMock(side_effect=[0, 0, 0.1, 11])
    monkeypatch.setattr(conftest, "time", SimpleNamespace(monotonic=monotonic))
    client = SimpleNamespace(_client=session)

    if task.get("related_document") or task.get("result_data", {}).get("document_id"):
        assert await conftest._upload_document(client, b"pdf") == 42
    else:
        with pytest.raises(RuntimeError, match="failed|without a document|after 10 s"):
            await conftest._upload_document(client, b"pdf")
    assert session.get.await_count == 2
    session.get.assert_awaited_with(
        "/api/tasks/", params={"task_id": "upload-task"}, timeout=2
    )
    assert sleep.await_count == (2 if task["status"] == "started" else 1)
