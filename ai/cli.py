#!/usr/bin/env python3
"""
CLI entrypoint for the AI post-processing service.

Modes:
    --once       Process all pending documents once and exit
    --watch      Poll continuously (default when run via Docker)
    --eval       Run offline evaluation against eval/golden_dataset.json
    --dry-run    Log what would happen without modifying any documents
    --purge-notes  Delete all AI-generated notes from previous runs

Usage:
    python cli.py --once
    python cli.py --watch
    python cli.py --eval
    python cli.py --once --dry-run
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


def _build_agent(config):
    """Instantiate the appropriate agent based on availability."""
    try:
        from agents.smart_graph_agent import SmartDocumentAgent
        log.info("Using SmartDocumentAgent (LangGraph)")
        return SmartDocumentAgent(config)
    except ImportError:
        from agents.legacy_agent import SeparatePipelineAgent
        log.info("LangGraph not available — falling back to SeparatePipelineAgent")
        return SeparatePipelineAgent(config)


async def main_async(args: argparse.Namespace) -> None:
    from core.config import AgentConfig
    from core.paperless import PaperlessClient
    from core.runner import purge_ai_notes, request_shutdown, run_batch
    from core.telemetry import setup_telemetry

    config = AgentConfig.from_env()

    if args.dry_run:
        config = config.model_copy(update={"dry_run": True})

    if not config.paperless_url:
        log.error("PAPERLESS_URL is not set")
        sys.exit(1)
    if not config.paperless_token:
        log.error("PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set")
        sys.exit(1)

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
    if config.dry_run:
        log.info("DRY RUN mode — no documents will be modified")

    with PaperlessClient(config.paperless_url, config.paperless_token) as client:
        # Verify Paperless connectivity
        log.info("Checking Paperless API connectivity...")
        try:
            import httpx
            from core.paperless import _raise_for_status
            r = client._client.get("/api/", follow_redirects=True)
            _raise_for_status(r)
            log.info(
                "Paperless API reachable (version: %s)",
                r.headers.get("x-version", "unknown"),
            )
        except Exception as e:
            log.error("Cannot reach Paperless API at %s: %s", config.paperless_url, e)
            sys.exit(1)

        # Verify LLM connectivity (warning only — GPU workstation may be off)
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
            purge_ai_notes(client, config.dry_run)
            return

        if args.eval:
            agent = _build_agent(config)
            from eval.run_evals import run_evals
            await run_evals(agent, config)
            return

        log.info("Resolving tag: '%s'", config.tag_pending)
        try:
            pending_id = client.get_tag_id(config.tag_pending, create=True)
        except Exception as e:
            log.error("Failed to resolve tag: %s", e)
            sys.exit(1)

        try:
            custom_field_id = client.get_or_create_custom_field(
                "ai_processed", data_type="date"
            )
            ai_result_field_id = client.get_or_create_custom_field(
                "ai_result", data_type="longtext"
            )
        except Exception as e:
            log.error("Failed to resolve custom fields: %s", e)
            sys.exit(1)

        log.info(
            "Tag ID: pending=%d | custom fields: ai_processed=%d ai_result=%d",
            pending_id,
            custom_field_id,
            ai_result_field_id,
        )

        agent = _build_agent(config)

        if args.once:
            success, failure = await run_batch(
                client, agent, config, pending_id, custom_field_id, ai_result_field_id
            )
            _write_heartbeat()
            log.info("Done. Success: %d, Failed: %d", success, failure)
        else:
            # Watch mode with graceful shutdown
            def _request_shutdown(signum: int, frame: object) -> None:
                log.info(
                    "Received %s — will stop after current document completes",
                    signal.Signals(signum).name,
                )
                request_shutdown()

            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT, _request_shutdown)

            from core.runner import is_shutdown_requested
            log.info(
                "Watch mode: polling every %ds (SIGTERM/Ctrl+C to stop gracefully)",
                config.poll_interval,
            )
            while not is_shutdown_requested():
                try:
                    success, failure = await run_batch(
                        client, agent, config, pending_id, custom_field_id, ai_result_field_id
                    )
                    if success or failure:
                        log.info("Batch done. Success: %d, Failed: %d", success, failure)
                except Exception as e:
                    log.error("Batch error: %s", e)
                _write_heartbeat()
                if is_shutdown_requested():
                    break
                log.info("Sleeping %ds...", config.poll_interval)
                try:
                    await asyncio.sleep(config.poll_interval)
                except asyncio.CancelledError:
                    break
            log.info("Shutdown complete.")


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
        help="Poll continuously (default when run via Docker)",
    )
    mode.add_argument(
        "--eval",
        action="store_true",
        help="Run offline evaluation against eval/golden_dataset.json",
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

    # Default to watch mode if neither flag is set
    if not args.once and not args.eval:
        args.once = False  # watch mode is the default

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
