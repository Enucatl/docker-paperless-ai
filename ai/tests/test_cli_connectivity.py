"""Exercise command dispatch through the real argument parser and entrypoint."""

import importlib.util
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def cli(monkeypatch):
    """Load the script with no inherited service or inference configuration."""
    spec = importlib.util.spec_from_file_location(
        "paperless_cli", Path(__file__).parents[1] / "cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with patch.dict(os.environ, {}, clear=True):
        monkeypatch.setattr("paperless_common.telemetry.setup_telemetry", MagicMock())
        yield module


@pytest.fixture
def paperless(cli, monkeypatch):
    """Mock Paperless HTTP transport, preserving the real client."""
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless.test")
    monkeypatch.setenv("PAPERLESS_TOKEN", "token")
    with patch("paperless_common.paperless.niquests.AsyncSession") as factory:
        session = AsyncMock()
        factory.return_value = session
        response = MagicMock(status_code=200, headers={"x-version": "5.1.2"})
        response.json.return_value = {"results": [], "next": None}
        session.get.return_value = response
        yield session


def test_eval_needs_no_paperless_or_default_models(cli, monkeypatch):
    """Offline evaluation reaches its runner with no unrelated services."""
    monkeypatch.setattr("sys.argv", ["cli.py", "--eval", "--split", "code-test"])
    with (
        patch(
            "paperless_ai.eval.run_evals.run_evals", new_callable=AsyncMock
        ) as evaluate,
        patch.object(cli, "PaperlessClient") as client,
        patch("paperless_ai.inference.complete", new_callable=AsyncMock) as complete,
    ):
        cli.main()
    evaluate.assert_awaited_once()
    assert evaluate.call_args.kwargs == {"split": "code-test"}
    assert evaluate.call_args.args[0].paperless_url == ""
    client.assert_not_called()
    complete.assert_not_awaited()


@pytest.mark.parametrize("dry_run", [False, True])
def test_purge_uses_only_paperless(cli, paperless, monkeypatch, caplog, dry_run):
    """Purge retains its dry-run flag and valid connectivity parameters."""
    monkeypatch.setattr(
        "sys.argv", ["cli.py", "--purge-notes"] + (["--dry-run"] if dry_run else [])
    )
    with (
        patch(
            "paperless_ai.core.runner.purge_ai_notes", new_callable=AsyncMock
        ) as purge,
        patch("paperless_ai.inference.complete", new_callable=AsyncMock) as complete,
    ):
        with caplog.at_level("INFO"):
            cli.main()
    paperless.get.assert_awaited_once_with("/api/")
    assert "5.1.2" in caplog.text
    assert purge.call_args.args[1] is dry_run
    complete.assert_not_awaited()


def test_paperless_failure_stops_command(cli, paperless, monkeypatch):
    """An unreachable API stops execution before note deletion."""
    monkeypatch.setattr("sys.argv", ["cli.py", "--purge-notes"])
    paperless.get.side_effect = ConnectionError("Connection refused")
    with patch(
        "paperless_ai.core.runner.purge_ai_notes", new_callable=AsyncMock
    ) as purge:
        with pytest.raises(SystemExit, match="1"):
            cli.main()
    purge.assert_not_awaited()


@pytest.mark.parametrize("dry_run", [False, True])
def test_cleanup_apply_needs_no_models(cli, paperless, monkeypatch, tmp_path, dry_run):
    """Load and apply a real empty plan without requiring inference settings."""
    from paperless_ai.core.correspondent_cleanup import (
        CorrespondentMergePlan,
        write_merge_plan,
    )

    plan_path = tmp_path / "plan.json"
    plan = CorrespondentMergePlan(
        1, "2026-09-29", "http://paperless.test", 0, 0, False, 0, [], [], []
    )
    write_merge_plan(plan, plan_path)
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "--cleanup-correspondents-apply", str(plan_path)]
        + (["--dry-run"] if dry_run else []),
    )
    with (
        patch(
            "paperless_ai.core.correspondent_cleanup.CleanupReviewStore.from_config",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("paperless_ai.inference.complete", new_callable=AsyncMock) as complete,
    ):
        cli.main()
    complete.assert_not_awaited()


@pytest.mark.parametrize("flag", ["--once", "--watch"])
def test_workers_require_metadata(cli, monkeypatch, flag):
    """Both worker modes reject missing models before opening services."""
    monkeypatch.setattr("sys.argv", ["cli.py", flag])
    with patch.object(cli, "PaperlessClient") as client:
        with pytest.raises(ValueError, match="INFERENCE_METADATA_MODEL is required"):
            cli.main()
    client.assert_not_called()


def test_once_retains_worker_initialization_without_chat(cli, paperless, monkeypatch):
    """Workers still probe metadata and initialize their shared resources."""
    monkeypatch.setenv("INFERENCE_METADATA_MODEL", "test-model")
    monkeypatch.setattr("sys.argv", ["cli.py", "--once"])
    monkeypatch.setattr(cli, "_write_heartbeat", MagicMock())
    with (
        patch("paperless_ai.inference.complete", new_callable=AsyncMock) as complete,
        patch(
            "paperless_ai.core.runtime.initialize_paperless",
            new_callable=AsyncMock,
            return_value=(1, 2, 3, 4),
        ) as initialize,
        patch("paperless_common.queue.TaskQueues", return_value=AsyncMock()) as queues,
        patch(
            "paperless_ai.core.runner.run_ocr_batch",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ) as ocr,
        patch(
            "paperless_ai.core.runner.run_metadata_batch",
            new_callable=AsyncMock,
            return_value=(0, 0),
        ) as metadata,
    ):
        cli.main()
    paperless.get.assert_awaited_once_with("/api/")
    assert complete.call_args.kwargs["domain"] == "startup_connectivity"
    initialize.assert_awaited_once()
    assert ocr.call_args.args[-1] == 4
    assert metadata.call_args.args[-3:] == (1, 2, 3)
    queues.return_value.close.assert_awaited_once()


@pytest.mark.parametrize("missing", ["PAPERLESS_URL", "PAPERLESS_TOKEN"])
def test_workers_require_paperless_credentials(cli, monkeypatch, missing):
    """Workers retain required Paperless credentials."""
    monkeypatch.setenv("INFERENCE_METADATA_MODEL", "test-model")
    monkeypatch.setenv("PAPERLESS_URL", "http://paperless.test")
    monkeypatch.setenv("PAPERLESS_TOKEN", "token")
    monkeypatch.delenv(missing)
    monkeypatch.setattr("sys.argv", ["cli.py", "--once"])
    with pytest.raises(SystemExit, match="1"):
        cli.main()


@pytest.mark.parametrize("mode", ["plan", "typesafe", "typesafe-plan"])
def test_cleanup_planning_needs_no_metadata_or_chat(
    cli, paperless, monkeypatch, tmp_path, mode
):
    """Planning writes its requested artifacts without unrelated model probes."""
    plan_path = tmp_path / "plan.json"
    analysis_dir = tmp_path / "analysis"
    args = ["cli.py"]
    if "plan" in mode:
        args += ["--cleanup-correspondents-plan", str(plan_path)]
    if "typesafe" in mode:
        args += ["--cleanup-typesafe", "--cleanup-analysis-dir", str(analysis_dir)]
        monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", args)
    with (
        patch(
            "paperless_ai.core.correspondent_cleanup.CleanupReviewStore.from_config",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("paperless_ai.inference.complete", new_callable=AsyncMock) as complete,
    ):
        cli.main()
    assert plan_path.exists() is ("plan" in mode)
    assert (analysis_dir / "scores.json").exists() is ("typesafe" in mode)
    complete.assert_not_awaited()


def test_default_service_config_still_requires_chat(cli, monkeypatch):
    """The combined HTTP service retains both model requirements."""
    monkeypatch.setenv("INFERENCE_METADATA_MODEL", "test-model")
    with pytest.raises(ValueError, match="INFERENCE_CHAT_MODEL is required"):
        cli.AgentConfig.from_env()
