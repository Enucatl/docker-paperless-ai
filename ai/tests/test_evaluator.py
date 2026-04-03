"""
Tests for the Phoenix-based evaluation runner (paperless_ai/eval/run_evals.py).

Coverage:
  - Entry filtering by split ("test", "validation", "all", "code-test")
  - Default split value for entries that omit the "split" field
  - Early return when no files survive filtering (before any Phoenix call)
  - DataFrame columns and row content passed to Phoenix
  - run_experiment called once per experiment defined in experiments.yaml
  - Fallback to get_dataset when create_dataset raises (already exists)
  - Error handling: missing dataset file, missing experiments YAML, Phoenix down
"""

import json
import pytest
import yaml
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import paperless_ai.eval.run_evals as _module
from paperless_ai.eval.run_evals import run_scientific_evaluation, run_evals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config():
    from paperless_ai.core.config import AgentConfig
    return AgentConfig(paperless_url="http://localhost:8000", paperless_token="dummy")


def _write_dataset(path: Path, entries: list) -> None:
    path.write_text(json.dumps({"entries": entries}))


def _write_experiments(path: Path, experiments: list | None = None) -> None:
    exps = experiments or [
        {"name": "exp1", "ocr_model": "gpt-4o", "metadata_model": "gpt-4o"},
    ]
    path.write_text(yaml.dump({"experiments": exps}))


def _existing_pdf(tmp_path: Path, name: str = "doc.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 placeholder")
    return p


class _Phoenix:
    """
    Context manager that stubs out every Phoenix import used inside
    run_scientific_evaluation so tests run without a live Phoenix instance.
    """

    def __init__(self):
        self.run_experiment = AsyncMock()
        self.client = AsyncMock()
        self.client.datasets.create_dataset = AsyncMock(return_value=MagicMock())
        self.client.datasets.get_dataset = AsyncMock(return_value=MagicMock())
        _client_cls = MagicMock(return_value=self.client)

        _exact_match = MagicMock()
        _exact_match.evaluate.return_value = [MagicMock(score=1.0)]

        self._sys_modules = {
            "phoenix": MagicMock(),
            "phoenix.client": MagicMock(AsyncClient=_client_cls),
            "phoenix.client.experiments": MagicMock(run_experiment=self.run_experiment),
            "phoenix.evals": MagicMock(LiteLLMModel=MagicMock(), llm_classify=MagicMock()),
            "phoenix.evals.metrics": MagicMock(exact_match=_exact_match),
        }
        self._patches = [
            patch.dict("sys.modules", self._sys_modules),
            patch("paperless_ai.core.telemetry.setup_telemetry"),
            patch("nest_asyncio.apply"),
        ]

    def __enter__(self):
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


# ---------------------------------------------------------------------------
# Early return: no Phoenix contact required
# ---------------------------------------------------------------------------


async def test_all_files_missing_returns_early(tmp_path):
    """When every entry references a non-existent file, return before touching Phoenix."""
    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(tmp_path / "ghost.pdf"), "split": "test"},
    ])
    _write_experiments(tmp_path / "exp.yaml")

    # No Phoenix mock — if Phoenix were contacted the test would hang/fail.
    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            await run_scientific_evaluation(_config())  # must not raise


async def test_split_mismatch_returns_early(tmp_path):
    """Requesting a split that has no entries in the dataset returns early."""
    _existing_pdf(tmp_path)  # file exists, but it's split="test"
    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(tmp_path / "doc.pdf"), "split": "test"},
    ])
    _write_experiments(tmp_path / "exp.yaml")

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            # Requesting "validation" — after split filter 0 entries remain → early return
            await run_scientific_evaluation(_config(), split="validation")


async def test_entries_without_split_field_default_to_test(tmp_path):
    """Entries that omit the 'split' key are treated as split='test'."""
    _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(tmp_path / "doc.pdf")},  # no "split" key
    ])
    _write_experiments(tmp_path / "exp.yaml")

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            # split="validation" → entry without split defaults to "test" → filtered out → early return
            await run_scientific_evaluation(_config(), split="validation")


# ---------------------------------------------------------------------------
# Split filtering correctness (verified via Phoenix call count / dataset rows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split,expected_count", [
    ("test",       1),
    ("validation", 1),
    ("all",        2),
])
async def test_split_filtering(tmp_path, split, expected_count):
    """Dataset uploaded to Phoenix contains only the entries matching the split."""
    test_pdf = _existing_pdf(tmp_path, "test.pdf")
    val_pdf  = _existing_pdf(tmp_path, "val.pdf")

    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(test_pdf), "split": "test"},
        {"file_path": str(val_pdf),  "split": "validation"},
    ])
    _write_experiments(tmp_path / "exp.yaml")

    captured_rows: list = []
    import pandas as pd
    real_df = pd.DataFrame

    def _capture(rows):
        captured_rows.extend(rows)
        return real_df(rows)

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix():
                with patch("pandas.DataFrame", side_effect=_capture):
                    await run_scientific_evaluation(_config(), split=split)

    assert len(captured_rows) == expected_count, (
        f"Expected {expected_count} rows for split={split!r}, got {len(captured_rows)}"
    )


async def test_code_test_split_selects_by_tag(tmp_path):
    """code-test split selects entries tagged 'code-test' regardless of their split field."""
    tagged_pdf    = _existing_pdf(tmp_path, "tagged.pdf")
    untagged_pdf  = _existing_pdf(tmp_path, "untagged.pdf")

    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(tagged_pdf),   "split": "validation", "tags": ["code-test"]},
        {"file_path": str(untagged_pdf), "split": "test"},
    ])
    _write_experiments(tmp_path / "exp.yaml")

    captured_rows: list = []
    import pandas as pd
    real_df = pd.DataFrame

    def _capture(rows):
        captured_rows.extend(rows)
        return real_df(rows)

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix():
                with patch("pandas.DataFrame", side_effect=_capture):
                    await run_scientific_evaluation(_config(), split="code-test")

    assert len(captured_rows) == 1
    assert "tagged.pdf" in captured_rows[0]["file_path"]


# ---------------------------------------------------------------------------
# DataFrame columns sent to Phoenix
# ---------------------------------------------------------------------------


async def test_dataset_dataframe_has_required_columns(tmp_path):
    """The DataFrame handed to Phoenix contains all six expected columns."""
    pdf = _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [
        {
            "file_path": str(pdf),
            "split": "test",
            "expected_correspondent": "Acme Corp",
            "expected_date": "2024-01-15",
            "expected_title_contains": "Invoice",
            "_verified_null_correspondent": False,
            "_verified_null_date": False,
        }
    ])
    _write_experiments(tmp_path / "exp.yaml")

    captured_rows: list = []
    import pandas as pd
    real_df = pd.DataFrame

    def _capture(rows):
        captured_rows.extend(rows)
        return real_df(rows)

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix():
                with patch("pandas.DataFrame", side_effect=_capture):
                    await run_scientific_evaluation(_config())

    assert captured_rows, "Expected at least one row"
    row = captured_rows[0]
    assert set(row.keys()) == {
        "file_path",
        "expected_correspondent",
        "expected_date",
        "expected_title_contains",
        "_verified_null_correspondent",
        "_verified_null_date",
    }
    assert row["expected_correspondent"] == "Acme Corp"
    assert row["expected_date"] == "2024-01-15"


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


async def test_run_experiment_called_once_per_yaml_experiment(tmp_path):
    """run_experiment is called exactly once per experiment entry in experiments.yaml."""
    pdf = _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [
        {"file_path": str(pdf), "split": "test"},
    ])
    _write_experiments(tmp_path / "exp.yaml", experiments=[
        {"name": "exp-a", "ocr_model": "gpt-4o", "metadata_model": "gpt-4o"},
        {"name": "exp-b", "ocr_model": "gpt-4o", "metadata_model": "gpt-4o"},
    ])

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix() as ph:
                await run_scientific_evaluation(_config())

    assert ph.run_experiment.call_count == 2


async def test_falls_back_to_get_dataset_when_create_fails(tmp_path):
    """If create_dataset raises (dataset exists), get_dataset is called instead."""
    pdf = _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [{"file_path": str(pdf), "split": "test"}])
    _write_experiments(tmp_path / "exp.yaml")

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix() as ph:
                ph.client.datasets.create_dataset.side_effect = Exception("already exists")
                await run_scientific_evaluation(_config())

    ph.client.datasets.get_dataset.assert_awaited_once()


# ---------------------------------------------------------------------------
# Experiment config: YAML overrides env, model fields reset
# ---------------------------------------------------------------------------


async def test_yaml_model_fields_override_env_config(tmp_path):
    """Model/endpoint fields in YAML completely replace env-config values."""
    pdf = _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [{"file_path": str(pdf), "split": "test"}])
    _write_experiments(tmp_path / "exp.yaml", experiments=[
        {"name": "custom", "ocr_model": "anthropic/claude-3-haiku", "metadata_model": "anthropic/claude-3-haiku"},
    ])

    env_config = _config()
    # env_config has ocr_model = "gemini/gemini-2.5-flash" by default
    built_configs: list = []

    real_AgentConfig = type(env_config)

    def _capture_config(**kwargs):
        cfg = real_AgentConfig(**kwargs)
        built_configs.append(cfg)
        return cfg

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with _Phoenix():
                with patch.object(_module, "AgentConfig", side_effect=_capture_config):
                    await run_scientific_evaluation(env_config)

    assert built_configs, "Expected at least one experiment config to be built"
    exp_cfg = built_configs[0]
    assert exp_cfg.ocr_model == "anthropic/claude-3-haiku"
    assert exp_cfg.name == "custom"


# ---------------------------------------------------------------------------
# Error handling: sys.exit paths
# ---------------------------------------------------------------------------


async def test_missing_golden_dataset_exits(tmp_path):
    """Missing GOLDEN_DATASET_PATH causes sys.exit(1)."""
    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "nonexistent.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with pytest.raises(SystemExit):
                await run_scientific_evaluation(_config())


async def test_missing_experiments_yaml_exits(tmp_path):
    """Missing EXPERIMENTS_YAML_PATH causes sys.exit(1)."""
    _write_dataset(tmp_path / "ds.json", [])
    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "nonexistent.yaml"):
            with pytest.raises(SystemExit):
                await run_scientific_evaluation(_config())


async def test_phoenix_unreachable_exits(tmp_path):
    """When Phoenix is unreachable, sys.exit(1) is called."""
    pdf = _existing_pdf(tmp_path)
    _write_dataset(tmp_path / "ds.json", [{"file_path": str(pdf), "split": "test"}])
    _write_experiments(tmp_path / "exp.yaml")

    # Stub only the imports — but make AsyncClient raise on instantiation
    def _fail_client(*args, **kwargs):
        raise ConnectionError("phoenix is down")

    mock_client_cls = MagicMock(side_effect=_fail_client)
    sys_modules = {
        "phoenix": MagicMock(),
        "phoenix.client": MagicMock(AsyncClient=mock_client_cls),
        "phoenix.client.experiments": MagicMock(run_experiment=AsyncMock()),
        "phoenix.evals": MagicMock(LiteLLMModel=MagicMock(), llm_classify=MagicMock()),
        "phoenix.evals.metrics": MagicMock(exact_match=MagicMock()),
    }

    with patch.object(_module, "GOLDEN_DATASET_PATH", tmp_path / "ds.json"):
        with patch.object(_module, "EXPERIMENTS_YAML_PATH", tmp_path / "exp.yaml"):
            with patch.dict("sys.modules", sys_modules):
                with patch("nest_asyncio.apply"):
                    with pytest.raises(SystemExit):
                        await run_scientific_evaluation(_config())


# ---------------------------------------------------------------------------
# run_evals wrapper
# ---------------------------------------------------------------------------


async def test_run_evals_delegates_to_run_scientific_evaluation():
    """run_evals is a thin wrapper: it calls run_scientific_evaluation with the right args."""
    with patch.object(_module, "run_scientific_evaluation", new=AsyncMock()) as mock_sci:
        config = _config()
        await run_evals(config, split="validation")

    mock_sci.assert_awaited_once_with(config, split="validation")
