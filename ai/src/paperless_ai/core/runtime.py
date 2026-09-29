"""Shared Paperless startup and polling worker lifecycle."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial

import niquests
from paperless_common.paperless import PaperlessClient, _raise_for_status
from paperless_common.queue import TaskQueues
from paperless_ai.core.config import AgentConfig
from paperless_ai.core import runner

log = logging.getLogger(__name__)


def _is_retryable_paperless_error(exc: Exception) -> bool:
    """Return whether a Paperless startup error may be transient."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is not None:
        return status_code in {408, 429} or status_code >= 500
    return isinstance(exc, niquests.RequestException)


async def initialize_paperless(
    client: PaperlessClient,
    config: AgentConfig,
    *,
    retry_delay: float = 1.0,
    max_retry_delay: float = 60.0,
) -> tuple[int, int, int, int]:
    """Wait for Paperless and ensure the AI-managed resources are ready."""
    attempt = 0
    while True:
        attempt += 1
        try:
            response = await client._client.get("/api/")
            _raise_for_status(response)
            log.info(
                "Paperless API reachable (version: %s)",
                response.headers.get("x-version", "unknown"),
            )

            if config.manage_paperless_workflows:
                added_wf_id, updated_wf_id = await client.ensure_ai_workflows(
                    tag_ocr=config.tag_ocr,
                    webhook_url=config.paperless_webhook_url,
                    webhook_secret=config.webhook_secret,
                )
                log.info(
                    "Paperless workflows ready: document_added=%d document_updated=%d",
                    added_wf_id,
                    updated_wf_id,
                )
            else:
                added_wf_id = updated_wf_id = 0

            custom_field_id = await client.get_or_create_custom_field(
                "ai_processed", data_type="date"
            )
            ai_summary_field_id = await client.get_or_create_custom_field(
                "ai_summary", data_type="longtext"
            )
            ai_result_field_id = await client.get_or_create_custom_field(
                "ai_result", data_type="longtext"
            )
            ai_ocr_output_field_id = await client.get_or_create_custom_field(
                "ai_ocr_output", data_type="longtext"
            )
            log.info(
                "Custom fields: ai_processed=%d ai_summary=%d ai_result=%d ai_ocr_output=%d",
                custom_field_id,
                ai_summary_field_id,
                ai_result_field_id,
                ai_ocr_output_field_id,
            )
            return (
                custom_field_id,
                ai_summary_field_id,
                ai_result_field_id,
                ai_ocr_output_field_id,
            )
        except Exception as exc:
            if not _is_retryable_paperless_error(exc):
                raise
            delay = min(max_retry_delay, retry_delay * 2 ** min(attempt - 1, 6))
            log.warning(
                "Paperless startup attempt %d failed: %s; retrying in %.1fs",
                attempt,
                exc,
                delay,
            )
            await asyncio.sleep(delay)


@asynccontextmanager
async def workers(
    client: PaperlessClient,
    config: AgentConfig,
    queues: TaskQueues,
    field_ids: tuple[int, int, int, int],
    heartbeat: Callable[[str], None],
    *,
    shutdown_timeout: float = 30.0,
):
    """Poll both stages and drain active batches before closing model probes."""
    processed, summary, result, ocr_output = field_ids

    async def poll(stage: str, batch: Callable[[], Awaitable[tuple[int, int]]]) -> None:
        """Run batches until stopped, including a heartbeat after failed batches."""
        while not runner.is_shutdown_requested():
            try:
                success, failure = await batch()
                if success or failure:
                    log.info("%s worker: %d ok / %d failed", stage, success, failure)
            except Exception:
                log.exception("%s worker error", stage)
            heartbeat(stage)
            await runner.wait_for_shutdown(config.poll_interval)

    batches = {
        "ocr": partial(runner.run_ocr_batch, client, config, queues, ocr_output),
        "metadata": partial(
            runner.run_metadata_batch,
            client,
            config,
            queues,
            processed,
            summary,
            result,
        ),
    }
    tasks = []
    for stage, batch in batches.items():
        heartbeat(stage)
        tasks.append(asyncio.create_task(poll(stage, batch), name=f"{stage}-worker"))
    try:
        yield tasks
    finally:
        runner.request_shutdown()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=shutdown_timeout
            )
        except TimeoutError:
            log.warning("Worker shutdown timed out; active batches cancelled")
        finally:
            await runner.close_model_probe_session()
