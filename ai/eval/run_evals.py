"""
Offline evaluation runner for document intelligence agents.

Runs the configured agent against eval/golden_dataset.json, grades outputs
with Phoenix ExactMatchEvaluator, and logs results to the local Phoenix UI.

Usage:
    python cli.py --eval
"""

import json
import logging
import os
import sys
import yaml
from pathlib import Path
from core.config import AgentConfig

log = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
EXPERIMENTS_YAML_PATH = Path(__file__).parent / "experiments.yaml"


async def run_scientific_evaluation(config: AgentConfig) -> None:
    """Run a parameter sweep across configurations defined in experiments.yaml."""
    try:
        import pandas as pd
        import phoenix as px
        from phoenix.evals import ExactMatchEvaluator, run_evals as px_run_evals
        from opentelemetry import trace
    except ImportError as e:
        log.error("Missing dependencies for evaluation: %s", e)
        sys.exit(1)

    if not GOLDEN_DATASET_PATH.exists():
        log.error("Golden dataset not found: %s", GOLDEN_DATASET_PATH)
        sys.exit(1)

    if not EXPERIMENTS_YAML_PATH.exists():
        log.error("Experiments YAML not found: %s", EXPERIMENTS_YAML_PATH)
        sys.exit(1)

    dataset = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    entries = dataset.get("entries", [])
    if not entries:
        log.warning("Golden dataset is empty")
        return

    # Load the YAML Experiments
    try:
        yaml_data = yaml.safe_load(EXPERIMENTS_YAML_PATH.read_text(encoding="utf-8"))
        experiments_raw = yaml_data.get("experiments", [])
    except Exception as e:
        log.error("Failed to parse experiments.yaml: %s", e)
        sys.exit(1)

    experiments = []
    for exp_dict in experiments_raw:
        # We start with the base CLI config and overlay the YAML experiment settings
        # This allows defaults like paperless_url/token to still come from env
        base_dict = config.model_dump()
        base_dict.update(exp_dict)
        experiments.append(AgentConfig(**base_dict))

    log.info("Starting scientific evaluation with %d experiments from YAML", len(experiments))
    from agents.smart_graph_agent import SmartDocumentAgent
    from core.telemetry import setup_telemetry

    setup_telemetry()
    tracer = trace.get_tracer(__name__)

    for exp_config in experiments:
        log.info("\n=== Running Experiment: %s ===", exp_config.name)
        log.info("Params: model=%s temp=%s reasoning=%s", 
                 exp_config.effective_metadata_model,
                 exp_config.temperature, 
                 exp_config.ocr_reasoning_effort)
        
        rows = []
        agent = SmartDocumentAgent(exp_config)

        # Wrap the whole experiment run in a parent span for easy grouping in Phoenix
        with tracer.start_as_current_span(f"Experiment: {exp_config.name}") as span:
            span.set_attribute("experiment.name", exp_config.name)
            span.set_attribute("llm.model", exp_config.effective_metadata_model)
            span.set_attribute("llm.temperature", exp_config.temperature if exp_config.temperature is not None else "default")
            span.set_attribute("llm.reasoning_effort", exp_config.ocr_reasoning_effort or "none")

            for entry in entries:
                file_path = entry["file_path"]
                if not Path(file_path).exists():
                    continue

                log.info("  Processing: %s", Path(file_path).name)
                try:
                    result = await agent.process(file_path, existing_hints={})
                    rows.append({
                        "file_path": file_path,
                        "expected_correspondent": entry.get("expected_correspondent"),
                        "actual_correspondent": result.metadata.correspondent,
                        "expected_date": entry.get("expected_date"),
                        "actual_date": result.metadata.document_date,
                        "expected_ocr_transcript": entry.get("expected_ocr_transcript"),
                        "actual_ocr_transcript": result.metadata.full_ocr_transcript,
                    })
                except Exception as e:
                    log.error("  Failed %s: %s", file_path, e)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        
        # Run evaluators
        from phoenix.evals import ExactMatchEvaluator, RougeScoreEvaluator
        
        corr_evaluator = ExactMatchEvaluator()
        ocr_evaluator = RougeScoreEvaluator()
        
        # Grade correspondent
        corr_results = px_run_evals(
            dataframe=df,
            evaluators=[corr_evaluator],
            provide_explanation=False,
            input_col="file_path", 
            output_col="actual_correspondent",
            label_col="expected_correspondent"
        )
        
        # Grade OCR transcript (if available in ground truth)
        ocr_results = None
        if "expected_ocr_transcript" in df.columns and not df["expected_ocr_transcript"].isna().all():
            ocr_results = px_run_evals(
                dataframe=df,
                evaluators=[ocr_evaluator],
                provide_explanation=False,
                input_col="file_path",
                output_col="actual_ocr_transcript",
                label_col="expected_ocr_transcript"
            )

        # Log to Phoenix
        try:
            phoenix_endpoint = os.environ.get(
                "PHOENIX_COLLECTOR_ENDPOINT",
                os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:6006"),
            )
            phoenix_base = phoenix_endpoint.split("/v1/")[0]
            client = px.Client(endpoint=phoenix_base)
            
            # Log evaluations with a unique name per experiment
            client.log_evaluations(
                corr_results[0].rename(columns={"label": "eval.correspondent_match"}),
                evaluation_name=f"Match ({exp_config.name})"
            )
            
            if ocr_results:
                client.log_evaluations(
                    ocr_results[0].rename(columns={"rouge_l": "eval.ocr_rouge_l"}),
                    evaluation_name=f"OCR Quality ({exp_config.name})"
                )
            
            log.info("Experiment '%s' results logged to Phoenix", exp_config.name)
        except Exception as e:
            log.warning("Could not log to Phoenix: %s", e)

    log.info("\nAll experiments complete! View results at http://localhost:6006")


async def run_evals(agent, config) -> None:
    """Wrapper to maintain CLI compatibility while enabling scientific mode."""
    await run_scientific_evaluation(config)
