# Paperless AI

## Python workflow

This repository has Python projects in `ai/`, `common/`, and `listener/`.
Run `uv` commands from the project being changed; the main test suite and its
lockfile are in `ai/`.

```bash
cd ai
uv sync --extra test --extra eval
uv run pytest tests/ -k "not test_webhook and not test_phase_b_pipeline and not test_search"
```

For Python code changes, run `uv run ruff format .` from the affected project
and the relevant tests.

## Integration tests

Tests that require Paperless, Redis, Qdrant, webhook delivery, or container
networking must use the Docker harness:

```bash
./run_tests.sh
./run_tests.sh --no-build
```

The harness builds the AI image, starts its dependencies, and removes the test
containers and anonymous volumes on exit. It sets
`MANAGE_PAPERLESS_WORKFLOWS=false`; webhook tests create and remove their own
workflows. Webhook tests must use the `webhook_session` fixture except when
verifying missing or invalid authentication.

## Evaluation

Run OCR and metadata experiments with Phoenix:

```bash
docker compose run --build --rm ai-eval --split code-test
```

Experiment definitions are in `ai/src/paperless_ai/eval/experiments.yaml`.
Use `code-test` for a quick check; `test`, `validation`, and `all` select the
corresponding dataset splits. If rebuilt datasets retain stale paths, run
`docker compose down -v phoenix` before retrying.

## Code map

- `ai/src/paperless_ai/search/`: indexing, retrieval, embeddings, Qdrant, and
  the search/webhook runtime.
- `ai/src/paperless_ai/core/`: configuration, runner, and Paperless client
  compatibility layer.
- `ai/src/paperless_ai/agents/`: OCR and metadata LangGraph pipelines.
- `ai/src/paperless_ai/eval/`: evaluation datasets, experiments, and metrics.
- `common/src/paperless_common/`: shared Paperless, queue, secrets, and
  telemetry helpers.
- `listener/src/paperless_listener/`: webhook ingress.
- `ai/tests/`: unit and Docker-backed integration tests.
