"""
Unit tests for the evaluation runner (eval/run_evals.py).

These tests do NOT require a running Paperless or Phoenix instance — they
exercise the agent invocation and grading logic using mocks and temp files.
"""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_result(title="Invoice", date="2024-01-15", correspondent="Acme Corp"):
    """Build a fake AgentResult for use in mock agents."""
    from agents.base import AgentResult, DocumentMetadata

    return AgentResult(
        metadata=DocumentMetadata(
            title=title,
            document_date=date,
            correspondent=correspondent,
            full_ocr_transcript="some text",
        ),
        elapsed_s=1.0,
        pages=1,
        chars=100,
        ocr_method="vision",
    )


class _MockAgent:
    """Fake agent that always returns a fixed AgentResult."""

    def __init__(self, result):
        self._result = result

    async def process(self, file_path, existing_hints):
        return self._result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_evals_skips_missing_files(tmp_path):
    """run_evals should skip entries where the file does not exist."""
    from core.config import AgentConfig

    dataset = {
        "entries": [
            {
                "file_path": str(tmp_path / "nonexistent.pdf"),
                "expected_correspondent": "Acme Corp",
                "expected_date": "2024-01-15",
            }
        ]
    }
    dataset_path = tmp_path / "golden_dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    config = AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="dummy",
    )
    agent = _MockAgent(_make_agent_result())

    # Patch the constant so run_evals uses our temp dataset
    import eval.run_evals as evals_module

    with patch.object(evals_module, "GOLDEN_DATASET_PATH", dataset_path):
        # Should return without error even though all files are missing
        with patch("eval.run_evals.pd") as mock_pd:
            mock_pd.DataFrame.return_value = MagicMock()
            await evals_module.run_evals(agent, config)
    # No assertion needed — the test passes if run_evals does not raise


@pytest.mark.asyncio
async def test_run_evals_grades_correct_prediction(tmp_path):
    """run_evals should produce correspondent_match=True for a correct prediction."""
    import pandas as pd

    from core.config import AgentConfig

    # Create a real 1-page PDF so the agent receives a real path
    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 700), "Invoice from Acme Corp")
        buf_path = tmp_path / "test.pdf"
        doc.save(str(buf_path))
        doc.close()
    except Exception:
        pytest.skip("PyMuPDF not available for this unit test")

    dataset = {
        "entries": [
            {
                "file_path": str(buf_path),
                "expected_correspondent": "Acme Corp",
                "expected_date": "2024-01-15",
                "expected_title_contains": "Invoice",
            }
        ]
    }
    dataset_path = tmp_path / "golden_dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    config = AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="dummy",
    )
    agent = _MockAgent(_make_agent_result(correspondent="Acme Corp", date="2024-01-15"))

    import eval.run_evals as evals_module

    rows_captured = []

    # Capture the DataFrame passed to pd.DataFrame so we can inspect results
    real_pd = pd

    def capture_df(rows):
        rows_captured.extend(rows)
        return real_pd.DataFrame(rows)

    with patch.object(evals_module, "GOLDEN_DATASET_PATH", dataset_path):
        # Mock out Phoenix so no network call is attempted
        mock_phoenix = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = [True]
        mock_phoenix.evals.ExactMatchEvaluator.return_value = mock_evaluator

        with (
            patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.evals": mock_phoenix.evals}),
            patch("eval.run_evals.pd.DataFrame", side_effect=capture_df),
        ):
            await evals_module.run_evals(agent, config)

    assert rows_captured, "Expected at least one evaluation row"
    row = rows_captured[0]
    assert row["actual_correspondent"] == "Acme Corp"
    assert row["actual_date"] == "2024-01-15"
    assert row["actual_title"] == "Invoice"


@pytest.mark.asyncio
async def test_run_evals_handles_agent_exception(tmp_path):
    """run_evals should skip entries where the agent raises an exception."""
    from core.config import AgentConfig

    try:
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        buf_path = tmp_path / "test.pdf"
        doc.save(str(buf_path))
        doc.close()
    except Exception:
        pytest.skip("PyMuPDF not available for this unit test")

    dataset = {
        "entries": [
            {
                "file_path": str(buf_path),
                "expected_correspondent": "Acme Corp",
                "expected_date": "2024-01-15",
            }
        ]
    }
    dataset_path = tmp_path / "golden_dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    class _BrokenAgent:
        async def process(self, file_path, existing_hints):
            raise ValueError("Simulated agent failure")

    config = AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="dummy",
    )

    import eval.run_evals as evals_module

    rows_captured = []

    import pandas as pd
    real_pd = pd

    def capture_df(rows):
        rows_captured.extend(rows)
        return real_pd.DataFrame(rows)

    with patch.object(evals_module, "GOLDEN_DATASET_PATH", dataset_path):
        mock_phoenix = MagicMock()
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = [False]
        mock_phoenix.evals.ExactMatchEvaluator.return_value = mock_evaluator

        with (
            patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.evals": mock_phoenix.evals}),
            patch("eval.run_evals.pd.DataFrame", side_effect=capture_df),
        ):
            await evals_module.run_evals(_BrokenAgent(), config)

    # When agent raises, no row is added (it's skipped with an error log)
    assert rows_captured == []


@pytest.mark.asyncio
async def test_run_evals_split_filtering(tmp_path):
    """run_evals should filter entries by split (test, validation, or all)."""
    import pandas as pd

    from core.config import AgentConfig

    # Create two PDFs
    try:
        import fitz
        for name in ["test1.pdf", "val1.pdf"]:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 700), f"Invoice {name}")
            buf_path = tmp_path / name
            doc.save(str(buf_path))
            doc.close()
    except Exception:
        pytest.skip("PyMuPDF not available for this unit test")

    dataset = {
        "entries": [
            {
                "file_path": str(tmp_path / "test1.pdf"),
                "expected_correspondent": "Acme Corp",
                "expected_date": "2024-01-15",
                "split": "test",
            },
            {
                "file_path": str(tmp_path / "val1.pdf"),
                "expected_correspondent": "Acme Corp",
                "expected_date": "2024-01-15",
                "split": "validation",
            },
        ]
    }
    dataset_path = tmp_path / "golden_dataset.json"
    dataset_path.write_text(json.dumps(dataset))

    # Create a minimal experiments.yaml
    experiments = {
        "experiments": [
            {"name": "test-exp", "ocr_model": "gpt-4o", "metadata_model": "gpt-4o", "temperature": 0.0}
        ]
    }
    experiments_path = tmp_path / "experiments.yaml"
    import yaml
    experiments_path.write_text(yaml.dump(experiments))

    config = AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="dummy",
    )
    agent = _MockAgent(_make_agent_result(correspondent="Acme Corp", date="2024-01-15"))

    import eval.run_evals as evals_module

    rows_captured = []

    real_pd = pd

    def capture_df(rows):
        rows_captured.extend(rows)
        return real_pd.DataFrame(rows)

    with patch.object(evals_module, "GOLDEN_DATASET_PATH", dataset_path):
        with patch.object(evals_module, "EXPERIMENTS_YAML_PATH", experiments_path):
            mock_phoenix = MagicMock()
            with (
                patch.dict("sys.modules", {"phoenix": mock_phoenix, "phoenix.evals": mock_phoenix.evals}),
                patch("eval.run_evals.pd.DataFrame", side_effect=capture_df),
            ):
                # Run with split="test" - should only process test1.pdf
                rows_captured.clear()
                await evals_module.run_evals(agent, config, split="test")
                assert len(rows_captured) == 1
                assert "test1.pdf" in rows_captured[0]["file_path"]

                # Run with split="validation" - should only process val1.pdf
                rows_captured.clear()
                await evals_module.run_evals(agent, config, split="validation")
                assert len(rows_captured) == 1
                assert "val1.pdf" in rows_captured[0]["file_path"]

                # Run with split="all" - should process both
                rows_captured.clear()
                await evals_module.run_evals(agent, config, split="all")
                assert len(rows_captured) == 2
