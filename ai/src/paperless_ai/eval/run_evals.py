"""Run metadata extraction experiments and Jev semantic evaluation."""

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
import pandas as pd

from paperless_ai.core.config import AgentConfig
from paperless_common.telemetry import setup_telemetry

log = logging.getLogger(__name__)

EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
EXPERIMENTS_YAML_PATH = Path(__file__).parent / "experiments.yaml"
PHOENIX_DATASET_BASE_NAME = "paperless-eval"


def _jev_score(output: Any, field: str) -> float:
    """Project one already-computed Jev score for Phoenix."""
    try:
        value = float((output or {}).get("_jev", {}).get(field, 0.0))
    except AttributeError, TypeError, ValueError:
        return 0.0
    return max(0.0, min(1.0, value))


def _phoenix_score(output: Any, field: str) -> dict[str, float]:
    """Return an explicit Phoenix numeric score payload."""
    return {"score": _jev_score(output, field)}


def jev_date(output: Any) -> dict[str, float]:
    """Project the Jev date probability as a Phoenix score."""
    return _phoenix_score(output, "date")


def jev_correspondent(output: Any) -> dict[str, float]:
    """Project the Jev correspondent probability as a Phoenix score."""
    return _phoenix_score(output, "correspondent")


def jev_title(output: Any) -> dict[str, float]:
    """Project the Jev title probability as a Phoenix score."""
    return _phoenix_score(output, "title")


def jev_metadata(output: Any) -> dict[str, float]:
    """Project the derived Jev arithmetic mean as a Phoenix score."""
    return _phoenix_score(output, "metadata")


def jev_date_confidence(output: Any) -> dict[str, float]:
    """Project Jev's date confidence as a Phoenix score."""
    return _phoenix_score(output, "date_confidence")


def jev_correspondent_confidence(output: Any) -> dict[str, float]:
    """Project Jev's correspondent confidence as a Phoenix score."""
    return _phoenix_score(output, "correspondent_confidence")


def jev_title_confidence(output: Any) -> dict[str, float]:
    """Project Jev's title confidence as a Phoenix score."""
    return _phoenix_score(output, "title_confidence")


def jev_metadata_confidence(output: Any) -> dict[str, float]:
    """Project mean Jev confidence as a Phoenix score."""
    return _phoenix_score(output, "metadata_confidence")


def _build_agent(exp_config: AgentConfig):
    """Instantiate the configured extraction agent."""
    module_path, class_name = exp_config.agent_class.rsplit(".", 1)
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)

    if class_name == "SmartDocumentAgent":
        from paperless_ai.agents.smart_graph_agent import _select_extraction_strategy

        strategy = _select_extraction_strategy(exp_config)
        log.info(
            "Experiment %s: using %s",
            exp_config.name,
            strategy.__class__.__name__,
        )
        return agent_class(exp_config, extraction_strategy=strategy)

    return agent_class(exp_config)


def _load_entries(path: Path, split: str) -> list[dict[str, Any]]:
    """Load input-only corpus entries for one requested split."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if split == "code-test":
        entries = [entry for entry in entries if "code-test" in entry.get("tags", [])]
    elif split != "all":
        entries = [entry for entry in entries if entry.get("split", "test") == split]
    return [{"file_path": entry["file_path"]} for entry in entries]


async def run_scientific_evaluation(config: AgentConfig, split: str = "test") -> None:
    """Run all configured extraction experiments against the eval corpus.

    Args:
        config: Base configuration, including the fixed TypeSafe settings.
        split: Corpus split to evaluate (``test``, ``validation``, ``all``, or
            ``code-test``).
    """
    try:
        from phoenix.client import AsyncClient
        from typesafe_sdk import TypeSafeClient
        from paperless_ai.eval.jev_evaluator import JevMetadataEvaluator
    except ImportError as error:
        log.error("Missing dependencies for evaluation: %s", error)
        sys.exit(1)

    if not EVAL_DATASET_PATH.exists():
        log.error("Evaluation corpus not found: %s", EVAL_DATASET_PATH)
        sys.exit(1)
    if not EXPERIMENTS_YAML_PATH.exists():
        log.error("Experiments YAML not found: %s", EXPERIMENTS_YAML_PATH)
        sys.exit(1)
    if not config.typesafe_api_key:
        log.error(
            "TYPESAFE_API_KEY (or TYPESAFE_API_KEY_FILE) is required for evaluation"
        )
        sys.exit(1)

    try:
        entries = _load_entries(EVAL_DATASET_PATH, split)
    except Exception as error:
        log.error("Failed to load evaluation corpus: %s", error)
        sys.exit(1)

    existing_entries = []
    for entry in entries:
        if Path(entry["file_path"]).exists():
            existing_entries.append(entry)
        else:
            log.debug("File not found: %s", entry.get("file_path"))

    if not existing_entries:
        log.warning("No local files found for split '%s'", split)
        return

    phoenix_endpoint = "http://phoenix:6006"
    phoenix_dataset_name = f"{PHOENIX_DATASET_BASE_NAME}-{split}"
    dataframe = pd.DataFrame(
        [{"file_path": entry["file_path"]} for entry in existing_entries]
    )

    try:
        phoenix_client = AsyncClient(base_url=phoenix_endpoint)
        try:
            phoenix_dataset = await phoenix_client.datasets.create_dataset(
                dataframe=dataframe,
                input_keys=["file_path"],
                name=phoenix_dataset_name,
                dataset_description=(
                    "Input-only document corpus; Jev scores are computed once "
                    "inside each experiment task."
                ),
            )
            log.info(
                "Created Phoenix input dataset '%s' with %d examples",
                phoenix_dataset_name,
                len(dataframe),
            )
        except Exception as create_error:
            try:
                phoenix_dataset = await phoenix_client.datasets.get_dataset(
                    dataset=phoenix_dataset_name
                )
                log.info("Using existing Phoenix dataset '%s'", phoenix_dataset_name)
            except Exception:
                raise create_error
        setup_telemetry()
    except Exception as error:
        log.error(
            "Cannot reach Phoenix at %s: %s\n"
            "Start it first with: docker compose up -d phoenix",
            phoenix_endpoint,
            error,
        )
        sys.exit(1)

    try:
        yaml_data = yaml.safe_load(EXPERIMENTS_YAML_PATH.read_text(encoding="utf-8"))
        experiments_raw = yaml_data.get("experiments", [])
    except Exception as error:
        log.error("Failed to parse experiments.yaml: %s", error)
        sys.exit(1)

    reset_for_experiments = {
        "name": None,
        "agent_class": "paperless_ai.agents.smart_graph_agent.SmartDocumentAgent",
        "ocr_model": "gemini/gemini-2.5-flash",
        "metadata_model": "gemini/gemini-2.5-flash",
        "chat_model": "gemini/gemini-2.5-flash",
        "ocr_endpoint": None,
        "metadata_endpoint": None,
        "chat_endpoint": None,
        "ocr_reasoning_effort": None,
        "metadata_reasoning_effort": None,
        "metadata_response_format": "auto",
        "ocr_temperature": None,
        "metadata_temperature": None,
        "ocr_max_image_dimension": None,
    }
    fixed_jev_settings = {
        "typesafe_api_key": config.typesafe_api_key,
        "typesafe_model": config.typesafe_model,
        "typesafe_endpoint": config.typesafe_endpoint,
    }

    experiments = []
    for experiment in experiments_raw:
        experiment_config = config.model_dump()
        experiment_config.update(reset_for_experiments)
        experiment_config.update(experiment)
        experiment_config.update(fixed_jev_settings)
        experiments.append(AgentConfig(**experiment_config))

    log.info(
        "Loaded %d eval experiments: %s; Jev model: %s",
        len(experiments),
        ", ".join(exp.name or "<unnamed>" for exp in experiments),
        config.typesafe_model,
    )

    for experiment_config in experiments:
        jev_client = None
        try:
            log.info(
                "=== Running experiment: %s (metadata model: %s) ===",
                experiment_config.name,
                experiment_config.metadata_model,
            )
            jev_client = TypeSafeClient(
                api_key=config.typesafe_api_key,
                model=config.typesafe_model,
                base_url=config.typesafe_endpoint,
            )
            jev_evaluator = JevMetadataEvaluator(jev_client, config.typesafe_model)
            agent = _build_agent(experiment_config)

            async def task(
                example,
                _agent=agent,
                _jev_evaluator=jev_evaluator,
                _experiment_config=experiment_config,
            ):
                """Extract metadata, then evaluate it once with Jev."""
                file_path = example.input["file_path"]
                result = await _agent.process(file_path, existing_hints={})
                metadata = result.metadata
                document_context = getattr(result, "metadata_context", "")
                if not document_context:
                    from paperless_ai.agents.smart_graph_agent import (
                        build_metadata_document_context,
                    )

                    document_context = build_metadata_document_context(
                        getattr(metadata, "full_ocr_transcript", "")
                    )

                evaluation = await _jev_evaluator.evaluate(
                    document_context=document_context,
                    title=metadata.title,
                    date=metadata.document_date,
                    correspondent=metadata.correspondent,
                )
                scores = {
                    "date": evaluation.date_score,
                    "correspondent": evaluation.correspondent_score,
                    "title": evaluation.title_score,
                    "metadata": evaluation.aggregate_score,
                    "date_confidence": evaluation.date_confidence,
                    "correspondent_confidence": evaluation.correspondent_confidence,
                    "title_confidence": evaluation.title_confidence,
                }
                scores["metadata_confidence"] = (
                    sum(
                        (
                            evaluation.date_confidence,
                            evaluation.correspondent_confidence,
                            evaluation.title_confidence,
                        )
                    )
                    / 3
                )

                jev_output = {
                    **scores,
                    "model": _experiment_config.typesafe_model,
                }

                return {
                    "correspondent": metadata.correspondent,
                    "date": metadata.document_date,
                    "title": metadata.title,
                    "_jev": jev_output,
                }

            await phoenix_client.experiments.run_experiment(
                dataset=phoenix_dataset,
                task=task,
                evaluators=[
                    jev_date,
                    jev_correspondent,
                    jev_title,
                    jev_metadata,
                    jev_date_confidence,
                    jev_correspondent_confidence,
                    jev_title_confidence,
                    jev_metadata_confidence,
                ],
                experiment_name=experiment_config.name,
                experiment_description=(
                    f"{experiment_config.agent_class.split('.')[-1]} | "
                    f"{experiment_config.metadata_model} | Jev {config.typesafe_model}"
                ),
                experiment_metadata={
                    "agent_class": experiment_config.agent_class,
                    "ocr_model": experiment_config.ocr_model,
                    "metadata_model": experiment_config.metadata_model,
                    "typesafe_model": config.typesafe_model,
                    "ocr_temperature": experiment_config.ocr_temperature,
                    "metadata_temperature": experiment_config.metadata_temperature,
                    "ocr_reasoning_effort": experiment_config.ocr_reasoning_effort,
                    "metadata_reasoning_effort": experiment_config.metadata_reasoning_effort,
                },
                concurrency=1,
                timeout=300,
            )
            log.info("Experiment '%s' complete", experiment_config.name)
        except Exception:
            log.exception("Experiment '%s' failed; continuing", experiment_config.name)
        finally:
            if jev_client is not None:
                jev_client.close()

    log.info("All experiments complete! View results at %s", phoenix_endpoint)


async def run_evals(config: AgentConfig, split: str = "test") -> None:
    """Run the Phoenix-based scientific evaluation."""
    await run_scientific_evaluation(config, split=split)
