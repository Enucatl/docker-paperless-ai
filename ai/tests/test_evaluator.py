"""Tests for the input-only Phoenix runner and Jev score projections."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

import paperless_ai.eval.run_evals as module
from paperless_ai.agents.base import AgentResult, DocumentMetadata
from paperless_ai.eval.run_evals import (
    jev_correspondent,
    jev_date,
    jev_metadata,
    jev_title,
    run_evals,
    run_scientific_evaluation,
)


def _config():
    """Build a minimal evaluation configuration."""
    from paperless_ai.core.config import AgentConfig

    return AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="dummy",
        metadata_model="test-metadata-model",
        chat_model="test-chat-model",
        typesafe_api_key="test-key",
        typesafe_model="jev-test",
    )


def _write_corpus(path: Path, entries: list[dict]) -> None:
    """Write a small input-only corpus fixture."""
    path.write_text(
        json.dumps(
            {
                "description": "Test corpus",
                "entries": entries,
            }
        )
    )


def _write_experiments(path: Path, count: int = 1) -> None:
    """Write extraction-only experiment fixtures."""
    experiments = [
        {
            "name": f"experiment-{index}",
            "ocr_model": "ocr-test",
            "metadata_model": "metadata-test",
        }
        for index in range(count)
    ]
    path.write_text(yaml.safe_dump({"experiments": experiments}))


def _existing_pdf(tmp_path: Path, name: str = "document.pdf") -> Path:
    """Create a path that passes the corpus file existence check."""
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4 placeholder")
    return path


class _Phoenix:
    """Stub Phoenix and execute each task once for assertions."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.run_experiment = AsyncMock(side_effect=self._run)
        self.client = AsyncMock()
        self.client.datasets.create_dataset = AsyncMock(return_value=MagicMock())
        self.client.datasets.get_dataset = AsyncMock(return_value=MagicMock())
        self.client.experiments.run_experiment = self.run_experiment
        self.outputs = []
        self.evaluators = []
        self._client_class = MagicMock(return_value=self.client)

    async def _run(self, **kwargs):
        """Run the supplied Phoenix task against one fake example."""
        self.evaluators.append(kwargs["evaluators"])
        output = await kwargs["task"](
            SimpleNamespace(input={"file_path": self.file_path})
        )
        self.outputs.append(output)
        return MagicMock()

    def patches(self):
        """Return patches for imports used by the runner."""
        return [
            patch.dict(
                "sys.modules",
                {
                    "phoenix": MagicMock(),
                    "phoenix.client": MagicMock(AsyncClient=self._client_class),
                },
            ),
            patch("paperless_common.telemetry.setup_telemetry"),
        ]


@pytest.mark.asyncio
async def test_corpus_is_uploaded_as_input_only(tmp_path):
    """Phoenix receives only the document identity, never annotations."""
    path = _existing_pdf(tmp_path)
    corpus = tmp_path / "eval.json"
    experiments = tmp_path / "experiments.yaml"
    _write_corpus(
        corpus,
        [
            {
                "file_path": str(path),
                "split": "test",
                "tags": ["code-test"],
                "non_eval_metadata": "kept out of Phoenix",
            }
        ],
    )
    _write_experiments(experiments)
    phoenix = _Phoenix(str(path))
    agent = MagicMock()
    agent.process = AsyncMock(
        return_value=AgentResult(
            metadata=DocumentMetadata(
                title="Invoice",
                document_date="2024-01-15",
                correspondent="Acme",
                full_ocr_transcript="OCR",
            ),
            metadata_context="OCR",
        )
    )
    typesafe_client = MagicMock()
    typesafe_client.system_one.return_value = SimpleNamespace(
        model="jev-test",
        nouls={
            "date": SimpleNamespace(noul=0.97),
            "correspondent": SimpleNamespace(noul=0.84),
            "title": SimpleNamespace(noul=0.91),
        },
    )

    with patch.object(module, "EVAL_DATASET_PATH", corpus):
        with patch.object(module, "EXPERIMENTS_YAML_PATH", experiments):
            with patch.object(module, "_build_agent", return_value=agent):
                with patch("typesafe_sdk.TypeSafeClient", return_value=typesafe_client):
                    patches = phoenix.patches()
                    for item in patches:
                        item.start()
                    try:
                        await run_scientific_evaluation(_config(), split="test")
                    finally:
                        for item in reversed(patches):
                            item.stop()

    dataset_call = phoenix.client.datasets.create_dataset.await_args
    dataframe = dataset_call.kwargs["dataframe"]
    assert list(dataframe.columns) == ["file_path"]
    assert dataset_call.kwargs["input_keys"] == ["file_path"]
    assert "output_keys" not in dataset_call.kwargs

    assert typesafe_client.system_one.call_count == 1
    output = phoenix.outputs[0]
    assert output["_jev"] == {
        "date": 0.97,
        "correspondent": 0.84,
        "title": 0.91,
        "metadata": pytest.approx((0.97 + 0.84 + 0.91) / 3),
        "model": "jev-test",
    }
    assert [e.__name__ for e in phoenix.evaluators[0]] == [
        "jev_date",
        "jev_correspondent",
        "jev_title",
        "jev_metadata",
    ]
    assert agent.process.await_args.kwargs == {"existing_hints": {}}

    request = typesafe_client.system_one.call_args.kwargs
    assert request["state"] == {
        "document": "OCR",
        "predicted_metadata": {
            "title": "Invoice",
            "date": "2024-01-15",
            "correspondent": "Acme",
        },
    }
    assert set(request["questions"]) == {"date", "correspondent", "title"}


@pytest.mark.asyncio
async def test_each_experiment_has_its_own_client(tmp_path):
    """Jev client state is not shared between extraction experiments."""
    path = _existing_pdf(tmp_path)
    corpus = tmp_path / "eval.json"
    experiments = tmp_path / "experiments.yaml"
    _write_corpus(corpus, [{"file_path": str(path), "split": "test"}])
    _write_experiments(experiments, count=2)
    phoenix = _Phoenix(str(path))
    agent = MagicMock()
    agent.process = AsyncMock(
        return_value=AgentResult(
            metadata=DocumentMetadata(
                title=None, document_date=None, correspondent=None
            ),
            metadata_context="same context",
        )
    )
    clients = [MagicMock(), MagicMock()]
    for client in clients:
        client.system_one.return_value = SimpleNamespace(
            model="jev-test",
            nouls={
                "date": SimpleNamespace(noul=0.1),
                "correspondent": SimpleNamespace(noul=0.2),
                "title": SimpleNamespace(noul=0.3),
            },
        )

    with patch.object(module, "EVAL_DATASET_PATH", corpus):
        with patch.object(module, "EXPERIMENTS_YAML_PATH", experiments):
            with patch.object(module, "_build_agent", return_value=agent):
                with patch("typesafe_sdk.TypeSafeClient", side_effect=clients):
                    patches = phoenix.patches()
                    for item in patches:
                        item.start()
                    try:
                        await run_scientific_evaluation(_config(), split="test")
                    finally:
                        for item in reversed(patches):
                            item.stop()

    assert all(client.system_one.call_count == 1 for client in clients)
    assert all(client.close.call_count == 1 for client in clients)


@pytest.mark.asyncio
async def test_jev_failure_is_logged_once_and_scores_zero(tmp_path, caplog):
    """A failed request is projected conservatively without evaluator retries."""
    path = _existing_pdf(tmp_path)
    corpus = tmp_path / "eval.json"
    experiments = tmp_path / "experiments.yaml"
    _write_corpus(corpus, [{"file_path": str(path), "split": "test"}])
    _write_experiments(experiments)
    phoenix = _Phoenix(str(path))
    agent = MagicMock()
    agent.process = AsyncMock(
        return_value=AgentResult(
            metadata=DocumentMetadata(title="Title"),
            metadata_context="context",
        )
    )
    typesafe_client = MagicMock()
    typesafe_client.system_one.side_effect = RuntimeError("service unavailable")

    with patch.object(module, "EVAL_DATASET_PATH", corpus):
        with patch.object(module, "EXPERIMENTS_YAML_PATH", experiments):
            with patch.object(module, "_build_agent", return_value=agent):
                with patch("typesafe_sdk.TypeSafeClient", return_value=typesafe_client):
                    patches = phoenix.patches()
                    for item in patches:
                        item.start()
                    try:
                        with caplog.at_level("ERROR", logger=module.log.name):
                            await run_scientific_evaluation(_config(), split="test")
                    finally:
                        for item in reversed(patches):
                            item.stop()

    assert typesafe_client.system_one.call_count == 1
    assert phoenix.outputs[0]["_jev"]["date"] == 0.0
    assert phoenix.outputs[0]["_jev"]["correspondent"] == 0.0
    assert phoenix.outputs[0]["_jev"]["title"] == 0.0
    assert (
        sum("Jev evaluation failed" in record.message for record in caplog.records) == 1
    )


def test_jev_projection_scores_are_continuous():
    """Phoenix projections preserve Jev probabilities instead of binarizing."""
    output = {"_jev": {"date": 0.97, "correspondent": 0.84, "title": 0.91}}

    assert jev_date(output) == {"score": 0.97}
    assert jev_correspondent(output) == {"score": 0.84}
    assert jev_title(output) == {"score": 0.91}
    assert jev_metadata({"_jev": {"metadata": 0.9066666667}}) == {
        "score": pytest.approx(0.9066666667)
    }
    assert jev_date({"_jev": {"date": "not-a-score"}}) == {"score": 0.0}


def test_phoenix_evaluator_wrapper_keeps_jev_probability_as_score():
    """Phoenix records the projected probability, not a binary label."""
    from phoenix.client.resources.experiments.evaluators import create_evaluator

    output = {"_jev": {"date": 0.97}}
    evaluator = create_evaluator()(jev_date)

    assert evaluator.evaluate(output=output) == {"score": 0.97}


@pytest.mark.asyncio
async def test_run_evals_wrapper():
    """The public wrapper forwards the selected split."""
    with patch.object(module, "run_scientific_evaluation", new=AsyncMock()) as runner:
        config = _config()
        await run_evals(config, split="validation")

    runner.assert_awaited_once_with(config, split="validation")
