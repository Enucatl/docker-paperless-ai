#!/usr/bin/env python3
"""
CLI entrypoint for the AI post-processing service.

Modes:
    --once       Process all pending documents once and exit
    --watch      Poll continuously with concurrent workers (default via Docker)
    --eval       Run offline evaluation against the input-only eval corpus
    --dry-run    Log what would happen without modifying any documents
    --purge-notes  Delete all AI-generated notes from previous runs
    --cleanup-correspondents-plan PATH   Write a reviewable correspondent merge plan
    --cleanup-correspondents-apply PATH  Apply an approved correspondent merge plan

Usage:
    python cli.py --once
    python cli.py --watch
    python cli.py --eval
    python cli.py --once --dry-run

Pipeline stages (tag-driven):
    ai:run-ocr      → OCR worker: download PDF, run document OCR, write content
    ai:run-metadata → Metadata worker: read content, run LLM, write title/date/correspondent
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

from paperless_ai.core.config import AgentConfig
from paperless_common.paperless import PaperlessClient, _raise_for_status


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


def _load_config(
    args: argparse.Namespace, *, metadata: bool = False, paperless: bool = True
) -> AgentConfig:
    """Load settings and validate the dependencies used by this command."""
    config = AgentConfig.from_env(required_models=("metadata",) if metadata else ())
    if args.dry_run:
        config = config.model_copy(update={"dry_run": True})
    if paperless:
        if not config.paperless_url:
            log.error("PAPERLESS_URL is not set")
            sys.exit(1)
        if not config.paperless_token:
            log.error("PAPERLESS_TOKEN (or PAPERLESS_TOKEN_FILE) is not set")
            sys.exit(1)
        from paperless_common.telemetry import setup_telemetry

        setup_telemetry(service_name=config.name, project_name=config.name)
    if config.dry_run:
        log.info("DRY RUN mode — no documents will be modified")
    return config


async def _check_paperless(client: PaperlessClient) -> None:
    """Fail early if a Paperless command cannot reach its API."""
    try:
        response = await client._client.get("/api/")
        _raise_for_status(response)
        log.info(
            "Paperless API reachable (version: %s)",
            response.headers.get("x-version", "unknown"),
        )
    except Exception as error:
        log.error("Cannot reach Paperless API: %s", error)
        sys.exit(1)


async def _run_eval(args: argparse.Namespace) -> None:
    """Run experiments with their own model settings, without Paperless."""
    from paperless_ai.eval.run_evals import run_evals

    await run_evals(_load_config(args, paperless=False), split=args.split)


async def _purge_notes(
    client: PaperlessClient, config: AgentConfig, args: argparse.Namespace
) -> None:
    """Remove AI notes using only Paperless."""
    from paperless_ai.core.runner import purge_ai_notes

    await purge_ai_notes(client, config.dry_run)


async def _cleanup_plan(
    client: PaperlessClient, config: AgentConfig, args: argparse.Namespace
) -> None:
    """Write a reviewable correspondent merge plan."""
    from paperless_ai.core.correspondent_cleanup import (
        CleanupReviewStore,
        build_correspondent_merge_plan,
        write_merge_plan,
        summarize_merge_plan,
        write_cleanup_analysis,
    )

    cleanup_review_store = await CleanupReviewStore.from_config(config)
    plan = await build_correspondent_merge_plan(
        client,
        config,
        typesafe=args.cleanup_typesafe,
        review_store=cleanup_review_store,
    )
    write_merge_plan(plan, args.cleanup_correspondents_plan)
    if args.cleanup_typesafe:
        analysis = write_cleanup_analysis(plan, args.cleanup_analysis_dir)
        log.info(
            "Wrote read-only TypeSafe analysis: %s, %s, %s",
            analysis["paths"]["csv"],
            analysis["paths"]["json"],
            analysis["paths"]["html"],
        )
    plan_summary = summarize_merge_plan(plan)
    log.info(
        "Wrote correspondent cleanup plan to %s",
        args.cleanup_correspondents_plan,
    )
    log.info(
        "Plan summary: clusters=%d orphan_deletes=%d candidate_pairs=%d merges=%d review=%d reject=%d planned_moves=%d",
        plan_summary["approved_clusters"],
        plan_summary["orphan_correspondents"],
        plan_summary["candidate_pairs"],
        plan_summary["merge_pairs"],
        plan_summary["review_pairs"],
        plan_summary["rejected_pairs"],
        plan_summary["planned_document_moves"],
    )


async def _cleanup_typesafe(
    client: PaperlessClient, config: AgentConfig, args: argparse.Namespace
) -> None:
    """Write read-only TypeSafe analysis."""
    from paperless_ai.core.correspondent_cleanup import (
        CleanupReviewStore,
        build_correspondent_merge_plan,
        write_cleanup_analysis,
    )

    cleanup_review_store = await CleanupReviewStore.from_config(config)
    plan = await build_correspondent_merge_plan(
        client, config, typesafe=True, review_store=cleanup_review_store
    )
    analysis = write_cleanup_analysis(plan, args.cleanup_analysis_dir)
    total = analysis["candidate_pairs"]
    counts = analysis["counts"]
    log.info(
        "TypeSafe cleanup analysis: correspondents=%d candidate_pairs=%d threshold=%.2f reject=%d (%.1f%%) review=%d (%.1f%%) merge=%d (%.1f%%)",
        plan.total_correspondents,
        total,
        config.correspondent_cleanup_candidate_threshold,
        counts["reject"],
        100 * counts["reject"] / total if total else 0,
        counts["review"],
        100 * counts["review"] / total if total else 0,
        counts["merge"],
        100 * counts["merge"] / total if total else 0,
    )
    log.info(
        "Wrote read-only analysis: %s, %s, %s",
        analysis["paths"]["csv"],
        analysis["paths"]["json"],
        analysis["paths"]["html"],
    )


async def _cleanup_apply(
    client: PaperlessClient, config: AgentConfig, args: argparse.Namespace
) -> None:
    """Apply an approved correspondent merge plan."""
    from paperless_ai.core.correspondent_cleanup import (
        CleanupReviewStore,
        apply_correspondent_merge_plan,
        load_merge_plan,
    )

    cleanup_review_store = await CleanupReviewStore.from_config(config)
    plan = load_merge_plan(args.cleanup_correspondents_apply)
    summary = await apply_correspondent_merge_plan(
        client,
        plan,
        dry_run=config.dry_run,
        review_store=cleanup_review_store,
    )
    log.info(
        "Correspondent cleanup applied from %s",
        args.cleanup_correspondents_apply,
    )
    log.info(
        "Apply summary: moved=%d deleted=%d skipped_clusters=%d skipped_orphans=%d skipped_nonempty_deletes=%d",
        summary["reassigned_documents"],
        summary["deleted_correspondents"],
        summary["skipped_clusters"],
        summary["skipped_orphans"],
        summary["skipped_nonempty_deletes"],
    )


async def _cleanup_weekly(args: argparse.Namespace) -> None:
    """Wait for queues, apply automatic cleanup, and publish the review plan."""
    from paperless_common.queue import TaskQueues
    from paperless_ai.core.correspondent_cleanup import (
        CleanupReviewStore,
        apply_correspondent_merge_plan,
        build_correspondent_merge_plan,
        write_merge_plan_atomically,
        wait_for_cleanup_queues,
    )

    config = _load_config(args)
    queues = TaskQueues(config.redis_url)
    try:
        log.info("Waiting for OCR and metadata queues to remain empty for 10 minutes")
        await wait_for_cleanup_queues(queues)
        cleanup_review_store = await CleanupReviewStore.from_config(config)
        async with PaperlessClient(
            config.paperless_url, config.paperless_token
        ) as client:
            plan = await build_correspondent_merge_plan(
                client, config, typesafe=True, review_store=cleanup_review_store
            )
            summary = await apply_correspondent_merge_plan(
                client,
                plan,
                review_store=cleanup_review_store,
                advance_manual_boundary=False,
            )
        if summary["skipped_nonempty_deletes"] or summary["skipped_stale_documents"]:
            raise RuntimeError("automatic cleanup became stale during apply")
        write_merge_plan_atomically(
            replace(
                plan,
                approved_clusters=[],
                orphan_correspondents=[],
                automatic_applied_clusters=plan.approved_clusters,
            ),
            args.cleanup_correspondents_weekly,
        )
        log.info("Weekly correspondent cleanup completed: %s", summary)
    finally:
        await queues.close()


async def _run_workers(
    client: PaperlessClient, config: AgentConfig, args: argparse.Namespace
) -> None:
    """Initialize and run the OCR and metadata workers."""
    from paperless_common.queue import TaskQueues
    from paperless_ai.core.runtime import initialize_paperless, workers
    from paperless_ai.core.runner import (
        clear_shutdown_request,
        request_shutdown,
        wait_for_shutdown,
        close_model_probe_session,
        run_ocr_batch,
        run_metadata_batch,
    )

    clear_shutdown_request()
    log.info("OCR layout-parsing endpoint: %s", config.ocr_endpoint)
    log.info("Pipeline tags: ocr=%r metadata=%r", config.tag_ocr, config.tag_metadata)
    log.info("Checking LLM connectivity (model: %s)...", config.metadata_model)
    try:
        _kwargs: dict = {
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 5,
        }
        from paperless_ai.inference import complete

        await complete(
            model=config.metadata_model,
            endpoint=config.metadata_endpoint,
            domain="startup_connectivity",
            **_kwargs,
        )
        log.info("LLM connectivity OK")
    except Exception as e:
        log.warning(
            "LLM connectivity check failed: %s — will retry during processing",
            e,
        )

    field_ids = await initialize_paperless(client, config)
    (
        custom_field_id,
        ai_summary_field_id,
        ai_result_field_id,
        ai_ocr_output_field_id,
    ) = field_ids

    # Set up Redis queues
    queues = TaskQueues(config.redis_url)
    log.info("Redis task queues: %s", config.redis_url)

    try:
        if args.once:
            # Sequential: OCR → metadata (docs flow through both stages in one run)
            ocr_s, ocr_f = await run_ocr_batch(
                client, config, queues, ai_ocr_output_field_id
            )
            meta_s, meta_f = await run_metadata_batch(
                client,
                config,
                queues,
                custom_field_id,
                ai_summary_field_id,
                ai_result_field_id,
            )
            _write_heartbeat()
            log.info(
                "Done. OCR: %d/%d  Metadata: %d/%d",
                ocr_s,
                ocr_s + ocr_f,
                meta_s,
                meta_s + meta_f,
            )
        else:
            # Watch mode: two concurrent workers, each polling their queue
            def _request_shutdown(signum: int, frame: object) -> None:
                log.info(
                    "Received %s — will stop after current document completes",
                    signal.Signals(signum).name,
                )
                request_shutdown()

            signal.signal(signal.SIGTERM, _request_shutdown)
            signal.signal(signal.SIGINT, _request_shutdown)

            log.info(
                "Watch mode: workers polling every %ds (SIGTERM/Ctrl+C to stop)",
                config.poll_interval,
            )

            async with workers(
                client, config, queues, field_ids, lambda stage: _write_heartbeat()
            ):
                await wait_for_shutdown()
            log.info("Shutdown complete.")
    finally:
        await close_model_probe_session()
        await queues.close()


async def main_async(args: argparse.Namespace) -> None:
    """Dispatch commands before initializing their service dependencies."""
    if args.cleanup_correspondents_weekly:
        await _cleanup_weekly(args)
        return
    if args.purge_notes:
        handler = _purge_notes
    elif args.eval:
        await _run_eval(args)
        return
    elif args.cleanup_correspondents_plan:
        handler = _cleanup_plan
    elif args.cleanup_typesafe:
        handler = _cleanup_typesafe
    elif args.cleanup_correspondents_apply:
        handler = _cleanup_apply
    else:
        handler = _run_workers

    config = _load_config(args, metadata=handler is _run_workers)
    async with PaperlessClient(config.paperless_url, config.paperless_token) as client:
        await _check_paperless(client)
        await handler(client, config, args)


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
    parser.add_argument(
        "--cleanup-typesafe",
        action="store_true",
        help="Use TypeSafe for read-only correspondent analysis",
    )
    parser.add_argument(
        "--cleanup-analysis-dir",
        default="correspondent-cleanup-analysis",
        help="Directory for read-only TypeSafe CSV, JSON, and HTML outputs",
    )
    mode.add_argument(
        "--watch",
        action="store_true",
        help="Poll continuously with two concurrent workers (default via Docker)",
    )
    mode.add_argument(
        "--eval",
        action="store_true",
        help="Run offline evaluation against the input-only eval corpus",
    )
    mode.add_argument(
        "--cleanup-correspondents-plan",
        metavar="PATH",
        help="Write a reviewable correspondent cleanup plan JSON and exit",
    )
    mode.add_argument(
        "--cleanup-correspondents-apply",
        metavar="PATH",
        help="Apply a previously generated correspondent cleanup plan JSON and exit",
    )
    mode.add_argument(
        "--cleanup-correspondents-weekly",
        metavar="PATH",
        help="Wait for queues, apply TypeSafe automatic cleanup, and publish review plan",
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

    if not (
        args.once
        or args.eval
        or args.cleanup_correspondents_plan
        or args.cleanup_correspondents_apply
        or args.cleanup_correspondents_weekly
        or args.cleanup_typesafe
    ):
        args.once = False  # watch mode is the default

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
