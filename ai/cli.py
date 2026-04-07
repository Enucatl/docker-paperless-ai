#!/usr/bin/env python3
"""
CLI entrypoint for the AI post-processing service.

Modes:
    --once       Process all pending documents once (all three stages) and exit
    --watch      Poll continuously with three concurrent workers (default via Docker)
    --eval       Run offline evaluation against eval/golden_dataset.json
    --dry-run    Log what would happen without modifying any documents
    --purge-notes  Delete all AI-generated notes from previous runs

Usage:
    python cli.py --once
    python cli.py --watch
    python cli.py --eval
    python cli.py --once --dry-run

Pipeline stages (tag-driven):
    ai:run-ocr      → OCR worker: download PDF, run vision OCR, write content
    ai:run-metadata → Metadata worker: read content, run LLM, write title/date/correspondent
    ai:run-embed    → Embed worker: read content+metadata, embed, upsert Qdrant
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

HEALTHCHECK_FILE = "/tmp/ai-healthy"


def _write_heartbeat() -> None:
    try:
        Path(HEALTHCHECK_FILE).write_text(str(time.time()))
    except OSError:
        pass


async def main_async(args: argparse.Namespace) -> None:
    from paperless_ai.core.config import AgentConfig
    from paperless_ai.core.paperless import PaperlessClient
    from paperless_ai.core.runner import (
        purge_ai_notes,
        request_shutdown,
        is_shutdown_requested,
        run_ocr_batch,
        run_metadata_batch,
        run_embed_batch,
    )
    from paperless_ai.core.telemetry import setup_telemetry
    from paperless_ai.search.embedder import EmbeddingAPIEmbedder
    from paperless_ai.search.queue import TaskQueues
    from paperless_ai.search.qdrant_store import QdrantDocumentStore

    config = AgentConfig.from_env()

    if args.dry_run:
        config = config.model_copy(update={"dry_run": True})

    if not config.paperless_url:
        log.error("PAPERLESS_URL is not set")
        sys.exit(1)
    if not config.paperless_token:
        log.error("PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set")
        sys.exit(1)

    if not args.eval:
        setup_telemetry()

    log.info("Paperless URL: %s", config.paperless_url)
    log.info(
        "OCR model: %s%s",
        config.ocr_model,
        f" (api_base={config.ocr_api_base})" if config.ocr_api_base else "",
    )
    log.info(
        "Metadata model: %s%s",
        config.effective_metadata_model,
        f" (api_base={config.metadata_api_base})" if config.metadata_api_base else "",
    )
    log.info("Embedding: %s @ %s", config.embedding_model, config.embedding_api_base)
    log.info(
        "Pipeline tags: ocr=%r metadata=%r embed=%r",
        config.tag_ocr, config.tag_metadata, config.tag_embed,
    )
    if config.dry_run:
        log.info("DRY RUN mode — no documents will be modified")

    async with PaperlessClient(config.paperless_url, config.paperless_token) as client:
        log.info("Checking Paperless API connectivity...")
        try:
            from paperless_ai.core.paperless import _raise_for_status
            r = await client._client.get("/api/")
            _raise_for_status(r)
            log.info(
                "Paperless API reachable (version: %s)",
                r.headers.get("x-version", "unknown"),
            )
        except Exception as e:
            log.error("Cannot reach Paperless API at %s: %s", config.paperless_url, e)
            sys.exit(1)

        # Skip LLM connectivity check for eval mode (experiments define their own models)
        if not args.eval:
            log.info("Checking LLM connectivity (model: %s)...", config.effective_metadata_model)
            try:
                import litellm
                _kwargs: dict = {
                    "model": config.effective_metadata_model,
                    "messages": [{"role": "user", "content": "Reply with OK"}],
                    "max_tokens": 5,
                }
                if config.metadata_api_base:
                    _kwargs["api_base"] = config.metadata_api_base
                await litellm.acompletion(**_kwargs)
                log.info("LLM connectivity OK")
            except Exception as e:
                log.warning(
                    "LLM connectivity check failed: %s — will retry during processing", e
                )

        if args.purge_notes:
            await purge_ai_notes(client, config.dry_run)
            return

        if args.eval:
            from paperless_ai.eval.run_evals import run_evals
            await run_evals(config, split=args.split)
            return

        if config.manage_paperless_workflows:
            try:
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
            except Exception as e:
                log.error("Failed to ensure Paperless workflows: %s", e)
                sys.exit(1)

        # Set up custom fields
        try:
            custom_field_id = await client.get_or_create_custom_field(
                "ai_processed", data_type="date"
            )
            ai_summary_field_id = await client.get_or_create_custom_field(
                "ai_summary", data_type="longtext"
            )
            ai_result_field_id = await client.get_or_create_custom_field(
                "ai_result", data_type="longtext"
            )
        except Exception as e:
            log.error("Failed to resolve custom fields: %s", e)
            sys.exit(1)

        log.info(
            "Custom fields: ai_processed=%d ai_summary=%d ai_result=%d",
            custom_field_id,
            ai_summary_field_id,
            ai_result_field_id,
        )

        # Set up three-stage Redis queues
        queues = TaskQueues(config.redis_url)
        log.info("Redis task queues: %s", config.redis_url)

        # Set up Qdrant store (optional — embedding skipped if unavailable)
        store = QdrantDocumentStore(config.qdrant_url)
        try:
            await store.ensure_collection()
            log.info("Qdrant collection ready (%s)", config.qdrant_url)
        except Exception as e:
            log.warning("Qdrant not reachable: %s — embedding will be skipped", e)
            store = None

        # Set up embeddings client (optional — embedding skipped if unavailable)
        embedder = EmbeddingAPIEmbedder(config.embedding_api_base, config.embedding_model)
        if not await embedder.check_connectivity():
            log.warning(
                "Embedding API not reachable at %s — embedding will be skipped",
                config.embedding_api_base,
            )
            await embedder.aclose()
            embedder = None

        try:
            if args.once:
                # Sequential: OCR → metadata → embed (docs flow through all stages in one run)
                ocr_s, ocr_f = await run_ocr_batch(client, config, queues)
                meta_s, meta_f = await run_metadata_batch(
                    client, config, queues, custom_field_id, ai_summary_field_id, ai_result_field_id
                )
                embed_s, embed_f = await run_embed_batch(client, config, queues, store, embedder)
                _write_heartbeat()
                log.info(
                    "Done. OCR: %d/%d  Metadata: %d/%d  Embed: %d/%d",
                    ocr_s, ocr_s + ocr_f,
                    meta_s, meta_s + meta_f,
                    embed_s, embed_s + embed_f,
                )
            else:
                # Watch mode: three concurrent workers, each polling their queue
                def _request_shutdown(signum: int, frame: object) -> None:
                    log.info(
                        "Received %s — will stop after current document completes",
                        signal.Signals(signum).name,
                    )
                    request_shutdown()

                signal.signal(signal.SIGTERM, _request_shutdown)
                signal.signal(signal.SIGINT, _request_shutdown)

                log.info(
                    "Watch mode: three workers polling every %ds (SIGTERM/Ctrl+C to stop)",
                    config.poll_interval,
                )

                async def _ocr_worker() -> None:
                    while not is_shutdown_requested():
                        try:
                            s, f = await run_ocr_batch(client, config, queues)
                            if s or f:
                                log.info("OCR worker: %d ok / %d failed", s, f)
                        except Exception as e:
                            log.error("OCR worker error: %s", e)
                        _write_heartbeat()
                        if is_shutdown_requested():
                            break
                        try:
                            await asyncio.sleep(config.poll_interval)
                        except asyncio.CancelledError:
                            break

                async def _metadata_worker() -> None:
                    while not is_shutdown_requested():
                        try:
                            s, f = await run_metadata_batch(
                                client, config, queues, custom_field_id, ai_summary_field_id, ai_result_field_id
                            )
                            if s or f:
                                log.info("Metadata worker: %d ok / %d failed", s, f)
                        except Exception as e:
                            log.error("Metadata worker error: %s", e)
                        if is_shutdown_requested():
                            break
                        try:
                            await asyncio.sleep(config.poll_interval)
                        except asyncio.CancelledError:
                            break

                async def _embed_worker() -> None:
                    while not is_shutdown_requested():
                        try:
                            s, f = await run_embed_batch(client, config, queues, store, embedder)
                            if s or f:
                                log.info("Embed worker: %d ok / %d failed", s, f)
                        except Exception as e:
                            log.error("Embed worker error: %s", e)
                        if is_shutdown_requested():
                            break
                        try:
                            await asyncio.sleep(config.poll_interval)
                        except asyncio.CancelledError:
                            break

                await asyncio.gather(_ocr_worker(), _metadata_worker(), _embed_worker())
                log.info("Shutdown complete.")
        finally:
            await queues.close()
            if embedder is not None:
                await embedder.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI post-processing for paperless-ngx documents"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--once",
        action="store_true",
        help="Process all pending documents once and exit",
    )
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously with three concurrent workers (default via Docker)",
    )
    mode.add_argument(
        "--eval",
        action="store_true",
        help="Run offline evaluation against eval/golden_dataset.json",
    )
    parser.add_argument(
        "--split",
        choices=["test", "validation", "all", "code-test"],
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without modifying any documents",
    )
    parser.add_argument(
        "--purge-notes",
        action="store_true",
        help="Delete all AI-generated notes from previous runs and exit",
    )
    args = parser.parse_args()

    if not args.once and not args.eval:
        args.once = False  # watch mode is the default

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
