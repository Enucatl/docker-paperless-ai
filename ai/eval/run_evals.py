"""
Offline evaluation runner for document intelligence agents.

Uses the Phoenix datasets & experiments API:
  1. Uploads the filtered golden dataset to Phoenix once per eval run.
  2. For each experiment defined in experiments.yaml, instantiates the
     configured agent and calls run_experiment() — Phoenix records
     per-example outputs and evaluator scores for side-by-side comparison.

Usage:
    python cli.py --eval [--split test|validation|all]
"""

import importlib
import json
import logging
import sys
import yaml
import nest_asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from core.config import AgentConfig
from core.telemetry import setup_telemetry

log = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
EXPERIMENTS_YAML_PATH = Path(__file__).parent / "experiments.yaml"
PHOENIX_DATASET_BASE_NAME = "paperless-golden"


async def run_scientific_evaluation(config: AgentConfig, split: str = "test") -> None:
    """
    Run a parameter sweep across configurations defined in experiments.yaml.

    Args:
        config: Base agent configuration (env defaults, overlaid by YAML).
        split: Dataset split to evaluate ("test", "validation", or "all").
    """
    try:
        import pandas as pd
        from phoenix.client import AsyncClient
        from phoenix.client.experiments import run_experiment
        from phoenix.evals.metrics import exact_match
    except ImportError as e:
        log.error("Missing dependencies for evaluation: %s", e)
        sys.exit(1)

    # run_experiment internally calls asyncio.run() for async tasks, which
    # would fail if we're already inside a running event loop (we are, via
    # asyncio.run() in cli.py). nest_asyncio patches the loop to allow this.
    nest_asyncio.apply()

    if not GOLDEN_DATASET_PATH.exists():
        log.error("Golden dataset not found: %s", GOLDEN_DATASET_PATH)
        sys.exit(1)

    if not EXPERIMENTS_YAML_PATH.exists():
        log.error("Experiments YAML not found: %s", EXPERIMENTS_YAML_PATH)
        sys.exit(1)

    # Load and filter golden dataset entries by split or tag.
    # "code-test" selects entries tagged ["code-test"] regardless of their split,
    # for quick single-file pipeline verification without running the full dataset.
    dataset_json = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    entries = dataset_json.get("entries", [])
    if split == "code-test":
        entries = [e for e in entries if "code-test" in e.get("tags", [])]
    elif split != "all":
        entries = [e for e in entries if e.get("split", "test") == split]
    entries = [e for e in entries if Path(e["file_path"]).exists()]
    if not entries:
        log.warning("No local files found for split '%s'", split)
        return

    phoenix_dataset_name = f"{PHOENIX_DATASET_BASE_NAME}-{split}"

    # Upload golden dataset to Phoenix. Phoenix stores it as a versioned
    # dataset; experiments reference it so results stay linked to the exact
    # ground truth used.
    df = pd.DataFrame([{
        "file_path": e["file_path"],
        "expected_correspondent": e.get("expected_correspondent"),
        "expected_date": e.get("expected_date"),
        "expected_title_contains": e.get("expected_title_contains"),
        "_verified_null_correspondent": e.get("_verified_null_correspondent", False),
        "_verified_null_date": e.get("_verified_null_date", False),
    } for e in entries])

    phoenix_endpoint = "http://phoenix:6006"
    try:
        phoenix_client = AsyncClient(base_url=phoenix_endpoint)
        try:
            phoenix_dataset = await phoenix_client.datasets.create_dataset(
                dataframe=df,
                input_keys=["file_path"],
                output_keys=[
                    "expected_correspondent", "expected_date", "expected_title_contains",
                    "_verified_null_correspondent", "_verified_null_date",
                ],
                name=phoenix_dataset_name,
            )
        except Exception:
            phoenix_dataset = await phoenix_client.datasets.get_dataset(dataset=phoenix_dataset_name)
            log.info("Using existing Phoenix dataset '%s'", phoenix_dataset_name)
        else:
            log.info("Uploaded %d examples to Phoenix dataset '%s'", len(df), phoenix_dataset_name)

        # Phoenix is reachable — enable OTel so LiteLLM spans carry token/cost data.
        setup_telemetry()
    except Exception as e:
        log.error(
            "Cannot reach Phoenix at %s: %s\n"
            "Start it first with: docker compose up -d phoenix",
            phoenix_endpoint, e,
        )
        sys.exit(1)

    # Load experiment configs, overlaying YAML fields on top of env defaults
    try:
        yaml_data = yaml.safe_load(EXPERIMENTS_YAML_PATH.read_text(encoding="utf-8"))
        experiments_raw = yaml_data.get("experiments", [])
    except Exception as e:
        log.error("Failed to parse experiments.yaml: %s", e)
        sys.exit(1)

    # Model/endpoint fields must be self-contained in the YAML — do not inherit
    # from the env config (which reflects the production watch-mode setup and
    # could route experiment calls to the wrong model or API base).
    # Operational fields (retries, concurrency, poll interval, etc.) are kept
    # from env so experiments benefit from the same runtime tuning.
    _reset_for_experiments = {
        "name": None,
        "agent_class": "agents.smart_graph_agent.SmartDocumentAgent",
        "ocr_model": "gemini/gemini-2.5-flash",
        "metadata_model": None,
        "ocr_api_base": None,
        "metadata_api_base": None,
        "ocr_reasoning_effort": None,
        "metadata_reasoning_effort": None,
        "temperature": None,
        "ocr_max_image_dimension": None,
        "llm_judge_model": "gemini/gemini-2.5-flash",
        "jury": None,
    }

    experiments = []
    for exp_dict in experiments_raw:
        base_dict = config.model_dump()
        base_dict.update(_reset_for_experiments)
        base_dict.update(exp_dict)
        experiments.append(AgentConfig(**base_dict))

    # --- Evaluators ---
    # Each function receives (output, expected) where:
    #   output   — dict returned by the task function (agent extraction result)
    #   expected — dict of output_keys from the Phoenix dataset (ground truth)
    # Return value is a float in [0, 1] (or bool, which Phoenix coerces).

    from eval.metrics import score_correspondent, score_date

    def _norm(v) -> str:
        return str(v or "").strip().lower()

    def date_exact(output, expected) -> float:
        """Case-insensitive exact match on ISO date strings."""
        return exact_match.evaluate({
            "output": _norm(output.get("date") if output else None),
            "expected": _norm(expected.get("expected_date")),
        })[0].score

    def correspondent_fuzzy(output, expected) -> float:
        """Token-sort fuzzy ratio after normalizing corporate suffixes and punctuation."""
        scores = score_correspondent(
            expected.get("expected_correspondent"),
            output.get("correspondent") if output else None,
        )
        return scores.get("corr_fuzzy_score", 0.0)

    def date_partial_credit(output, expected) -> float:
        """Linear partial credit: 1.0 at exact match, 0.0 at 365+ days off."""
        scores = score_date(
            expected.get("expected_date"),
            output.get("date") if output else None,
        )
        return scores.get("date_partial_credit", 0.0)

    # LLM-as-a-jury: classify the extracted title as appropriate or not,
    # using the OCR transcript as context. Avoids the need for per-document
    # reference titles — judges decide based on content relevance alone.
    #
    # When exp_config.jury is set, each member votes independently and the
    # result is determined by majority vote, which reduces single-model bias
    # and improves alignment with human judgment. Ties (even-sized jury with
    # equal split) are resolved as "inappropriate" (score 0.0) to keep the
    # metric conservative.
    # If no jury is configured, a single judge uses llm_judge_model.
    _TITLE_JUDGE_TEMPLATE = """
You are evaluating a document title extracted by an AI system.

Document text (excerpt):
{ocr_transcript}

Extracted title: {title}

Is this title appropriate for the document?
An appropriate title must be:
- Accurate — reflects the actual content of the document
- Specific — not generic placeholders like "Document", "Letter", or "Untitled"
- Descriptive — gives a reader a clear idea of what the document is about

Respond with exactly one word: "appropriate" or "inappropriate".
""".strip()

    from phoenix.evals import llm_classify, LiteLLMModel

    def _safe_majority_vote(votes: list[str]) -> str:
        """Return the majority label, or 'tie' when no label has a strict majority."""
        count = Counter(votes)
        most_common = count.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]
        return "tie"

    def _run_judge(llm_model: LiteLLMModel, row_df) -> str:
        """Run a single judge on a one-row DataFrame; returns the label or 'error'."""
        try:
            result = llm_classify(
                dataframe=row_df,
                template=_TITLE_JUDGE_TEMPLATE,
                model=llm_model,
                rails=["appropriate", "inappropriate"],
                provide_explanation=False,
            )
            return result["label"].iloc[0]
        except Exception as e:
            log.warning("Jury member %s failed: %s", llm_model.model, e)
            return "error"

    # --- Agent factory ---
    def _build_agent(exp_config: AgentConfig):
        module_path, class_name = exp_config.agent_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        # For SmartDocumentAgent, auto-detect and select extraction strategy
        if class_name == "SmartDocumentAgent":
            from agents.smart_graph_agent import _select_extraction_strategy
            strategy = _select_extraction_strategy(exp_config)
            log.info("Experiment %s: Using %s", exp_config.name, strategy.__class__.__name__)
            return agent_class(exp_config, extraction_strategy=strategy)

        return agent_class(exp_config)

    # --- Run one experiment per config ---
    for exp_config in experiments:
        log.info("\n=== Running Experiment: %s ===", exp_config.name)
        log.info(
            "Params: agent=%s model=%s temp=%s ocr_reasoning=%s metadata_reasoning=%s",
            exp_config.agent_class,
            exp_config.effective_metadata_model,
            exp_config.temperature,
            exp_config.ocr_reasoning_effort,
            exp_config.metadata_reasoning_effort,
        )

        # Build LiteLLMModel instances for each jury member (or single judge).
        # Each experiment may configure a different jury, so this is built per-run.
        if exp_config.jury:
            jury_models = [
                LiteLLMModel(
                    model=member.model,
                    temperature=member.temperature if member.temperature is not None else 0.0,
                    model_kwargs=member.to_litellm_model_kwargs(),
                )
                for member in exp_config.jury
            ]
            log.info(
                "Title jury: %d judges — %s",
                len(jury_models),
                ", ".join(m.model for m in exp_config.jury),
            )
        else:
            jury_models = [LiteLLMModel(model=exp_config.llm_judge_model)]
            log.info("Title judge: single model — %s", exp_config.llm_judge_model)

        # _models=jury_models captures the current jury in the closure.
        def title_llm_jury(output, _models=jury_models) -> float:
            """Jury vote: 1.0 if the majority of judges find the title appropriate.

            Each judge runs via llm_classify on a one-row DataFrame. Judges run in
            parallel via ThreadPoolExecutor. The majority label wins; ties and
            all-error cases resolve to 0.0 (conservative / "inappropriate").
            """
            if not output or not output.get("title"):
                return 0.0
            row_df = pd.DataFrame([{
                "ocr_transcript": (output.get("ocr_transcript") or "")[:3000],
                "title": output.get("title"),
            }])
            with ThreadPoolExecutor(max_workers=len(_models)) as pool:
                votes = list(pool.map(lambda m: _run_judge(m, row_df), _models))
            valid_votes = [v for v in votes if v in ("appropriate", "inappropriate")]
            if not valid_votes:
                return 0.0
            verdict = _safe_majority_vote(valid_votes)
            return 1.0 if verdict == "appropriate" else 0.0

        EVALUATORS = [
            date_exact,
            correspondent_fuzzy,
            date_partial_credit,
            title_llm_jury,
        ]

        agent = _build_agent(exp_config)

        # _agent=agent captures the current agent in the closure — without the
        # default arg, all iterations would share the last loop value.
        async def task(example, _agent=agent):
            file_path = example.input["file_path"]
            result = await _agent.process(file_path, existing_hints={})
            return {
                "correspondent": result.metadata.correspondent,
                "date": result.metadata.document_date,
                "title": result.metadata.title,
                "ocr_transcript": result.metadata.full_ocr_transcript,
            }

        try:
            await run_experiment(
                dataset=phoenix_dataset,
                task=task,
                evaluators=EVALUATORS,
                experiment_name=exp_config.name,
                experiment_description=(
                    f"{exp_config.agent_class.split('.')[-1]} | {exp_config.effective_metadata_model}"
                ),
                experiment_metadata={
                    "agent_class": exp_config.agent_class,
                    "ocr_model": exp_config.ocr_model,
                    "metadata_model": exp_config.effective_metadata_model,
                    "temperature": exp_config.temperature,
                    "ocr_reasoning_effort": exp_config.ocr_reasoning_effort,
                    "metadata_reasoning_effort": exp_config.metadata_reasoning_effort,
                },
                client=phoenix_client,
                timeout=300,
            )
            log.info("Experiment '%s' complete", exp_config.name)
        except Exception as e:
            log.error("Experiment '%s' failed: %s", exp_config.name, e)

    log.info("\nAll experiments complete! View results at %s", phoenix_endpoint)


async def run_evals(agent, config, split: str = "test") -> None:
    """
    Wrapper to maintain CLI compatibility while enabling scientific mode.

    Args:
        agent: Document processing agent (unused but kept for CLI compat)
        config: Agent configuration
        split: Dataset split to evaluate ("test", "validation", or "all")
    """
    await run_scientific_evaluation(config, split=split)
