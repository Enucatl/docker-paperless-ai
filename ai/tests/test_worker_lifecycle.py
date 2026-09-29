"""Worker shutdown wakes polling and allows active writes to finish."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from paperless_ai.core import runner, runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [False, True])
async def test_shutdown_wakes_idle_workers_and_drains_active_batch(monkeypatch, active):
    """Stopping never starts another batch or cancels an active write."""
    runner.clear_shutdown_request()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    heartbeats = []

    async def batch(*args):
        """Hold one batch until shutdown has been requested."""
        started.set()
        if active:
            await release.wait()
        finished.set()
        return 1, 0

    ocr = AsyncMock(side_effect=batch)
    close = AsyncMock()
    monkeypatch.setattr(runner, "run_ocr_batch", ocr)
    monkeypatch.setattr(runner, "run_metadata_batch", AsyncMock(return_value=(0, 0)))
    monkeypatch.setattr(runner, "close_model_probe_session", close)

    async def serve():
        """Mirror the HTTP lifespan's context exit."""
        async with runtime.workers(
            None,
            SimpleNamespace(poll_interval=3600),
            None,
            (1, 2, 3, 4),
            heartbeats.append,
        ) as tasks:
            await started.wait()
        return tasks

    service = asyncio.create_task(serve())
    await started.wait()
    await asyncio.sleep(0)
    assert runner.is_shutdown_requested()
    if active:
        assert not service.done()
        assert not finished.is_set()
        release.set()
    tasks = await asyncio.wait_for(service, timeout=1)
    assert finished.is_set()
    assert all(task.done() and not task.cancelled() for task in tasks)
    ocr.assert_awaited_once_with(None, SimpleNamespace(poll_interval=3600), None, 4)
    assert heartbeats.count("ocr") == 2
    assert "metadata" in heartbeats
    close.assert_awaited_once()
    runner.clear_shutdown_request()


@pytest.mark.asyncio
async def test_worker_errors_heartbeat_and_shutdown_is_bounded(monkeypatch):
    """A stuck batch is cancelled after grace; errors still update health."""
    runner.clear_shutdown_request()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    heartbeats = []

    async def stuck(*args):
        """Simulate an unresponsive inference call."""
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(runner, "run_ocr_batch", stuck)
    monkeypatch.setattr(
        runner, "run_metadata_batch", AsyncMock(side_effect=ValueError("failed"))
    )
    monkeypatch.setattr(runner, "close_model_probe_session", AsyncMock())
    async with runtime.workers(
        None,
        SimpleNamespace(poll_interval=3600),
        None,
        (1, 2, 3, 4),
        heartbeats.append,
        shutdown_timeout=0.01,
    ):
        await started.wait()
    assert cancelled.is_set()
    assert heartbeats.count("metadata") == 2
    runner.clear_shutdown_request()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_http_lifespan_stops_workers_and_closes_resources(monkeypatch, fail):
    """HTTP exit drains workers and releases clients even on application errors."""
    from paperless_ai.search import chat_store, webhook

    config = SimpleNamespace(name="test", poll_interval=3600)
    client = SimpleNamespace(aclose=AsyncMock())
    queues = SimpleNamespace(close=AsyncMock())
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless")
    monkeypatch.setenv("CHAT_DATABASE_URL", "postgresql://test")
    monkeypatch.setattr(webhook, "read_secret", lambda name: "secret")
    monkeypatch.setattr(webhook.AgentConfig, "from_env", lambda: config)
    monkeypatch.setattr(webhook, "PaperlessClient", lambda *args: client)
    monkeypatch.setattr(webhook, "TaskQueues", lambda *args: queues)
    monkeypatch.setattr(webhook, "setup_telemetry", lambda **kwargs: None)
    monkeypatch.setattr(
        chat_store,
        "ChatStore",
        lambda *args, **kwargs: SimpleNamespace(migrate=AsyncMock()),
    )
    initialize = AsyncMock(return_value=(1, 2, 3, 4))
    monkeypatch.setattr(webhook, "initialize_paperless", initialize)
    monkeypatch.setattr(runner, "run_ocr_batch", AsyncMock(return_value=(0, 0)))
    monkeypatch.setattr(runner, "run_metadata_batch", AsyncMock(return_value=(0, 0)))
    monkeypatch.setattr(runner, "close_model_probe_session", AsyncMock())

    async def serve():
        """Enter the actual HTTP lifespan without external services."""
        async with webhook.lifespan(webhook.app):
            assert webhook._worker_ready
            assert all(webhook._worker_heartbeats.values())
            await asyncio.sleep(0)
            if fail:
                raise ValueError("application failed")

    if fail:
        with pytest.raises(ValueError, match="application failed"):
            await asyncio.wait_for(serve(), timeout=1)
    else:
        await asyncio.wait_for(serve(), timeout=1)
    initialize.assert_awaited_once_with(client, config)
    client.aclose.assert_awaited_once()
    queues.close.assert_awaited_once()
    assert not webhook._worker_ready
    assert not webhook._worker_tasks
    runner.clear_shutdown_request()
