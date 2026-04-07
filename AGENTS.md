# Development with Claude Code Agents

## Dependency Management

Uses `uv` for fast, reliable Python dependency management:

```bash
uv sync              # Install dependencies in virtual environment
uv run python ...    # Run Python with virtual environment activated
uv run pytest ...    # Run pytest with virtual environment activated
```

This avoids managing `.venv` manually and ensures consistent builds.

## Running Tests

**Local unit tests** (no infrastructure):
```bash
cd ai
uv sync --extra test --extra eval
uv run pytest tests/ -k "not test_webhook and not test_phase_b_pipeline and not test_search"
```

Notes:
- Run `uv` commands from the `ai/` directory. The Python project and lockfile live there.
- Use local `uv run pytest` for fast feedback on pure unit tests and small targeted test files.
- Tests that need Paperless, Redis, Qdrant, webhook delivery, or real container networking should be treated as Docker tests even if they look small.

**Full E2E tests** (requires Docker):
```bash
./run_tests.sh                # Full E2E test suite
./run_tests.sh --no-build     # Skip rebuild (faster re-runs)
```

The test harness:
1. Builds the Docker image for the AI service
2. Spins up all infrastructure (Paperless, Qdrant, Redis, webhook-listener)
3. Runs pytest inside the AI container
4. Tears down all containers and anonymous volumes on exit

Test-specific workflow note:
- `docker-compose.test.yml` sets `MANAGE_PAPERLESS_WORKFLOWS=false`.
- Production startup auto-manages the Paperless workflows, but webhook integration tests create and delete their own workflows and must remain authoritative.

**Important:** Full E2E tests require Docker infrastructure. Use `uv run pytest` for unit-level testing without Docker. `run_tests.sh` is the authoritative check for integration failures.

**API Compatibility Coverage:** Tests cover external library API usage (e.g., niquests AsyncSession methods, Redis async methods) to catch parameter incompatibilities early. This prevents runtime errors like `follow_redirects` (httpx) being used with niquests.

**New API compatibility test suite:**
- `test_paperless_client.py` — Tests PaperlessClient async context manager and correct niquests usage
- `test_embedder_client.py` — Tests InfinityEmbedder async context manager and connectivity checks
- `test_cli_connectivity.py` — Tests CLI Paperless API connectivity checks without invalid parameters
- `test_niquests_api_compatibility.py` — Documents niquests vs httpx differences (close() vs aclose(), allow_redirects vs follow_redirects)

These tests prevent regressions like:
- Using `follow_redirects=True` (httpx parameter) with niquests (which uses `allow_redirects`)
- Calling `session.aclose()` instead of `session.close()` on niquests.AsyncSession
- Using `content=` (httpx parameter) instead of `data=` for raw bytes in niquests

**HTTP client policy:** niquests is the sole HTTP client in this project — do not introduce httpx as a dependency or in test code. niquests is a fork of requests with async support (`AsyncSession`) and is API-compatible with requests, not httpx.

**Webhook auth in tests:** `WEBHOOK_SECRET=test-secret-key-12345` is always set in `docker-compose.test.yml`. All tests that POST to `/webhook/document` must use the `webhook_session` fixture (which carries the auth header). Tests specifically verifying auth rejection (`test_webhook_rejects_missing_token`, `test_webhook_rejects_wrong_token`) use their own bare `niquests.AsyncSession()` without the token. Do not use in-process ASGI transport for auth tests — the container is the source of truth.

## Running Evaluation Experiments

Evaluate OCR and metadata extraction with different LLM configurations via Phoenix:

```bash
docker compose run --build --rm ai-eval --split code-test
```

Notes:
- `ai-eval` is a separate Docker service/image from `ai` so production and eval dependencies do not conflict.
- Production tracing to Phoenix still comes from the regular `ai` service via OTLP; the separate eval image only carries the offline-evaluation stack.

Flags:
- `--split code-test` — Quick pipeline verification using tagged examples (fastest)
- `--split test` — Full test set evaluation
- `--split validation` — Validation set evaluation  
- `--split all` — All examples

The evaluation framework:
1. Loads experiment configs from `ai/src/paperless_ai/eval/experiments.yaml`
2. Instantiates each configured agent (OCR model, metadata model, parameters)
3. Runs the agent on golden dataset examples
4. Evaluates outputs via Phoenix (metrics: date exact/partial, correspondent fuzzy, title jury vote)
5. Publishes results to Phoenix dashboard at `http://phoenix:6006`

**Configuring experiments:** Edit `experiments.yaml` to add/modify configurations. Supports:
- Different OCR and metadata models
- Custom API bases for local vLLM endpoints
- Extra parameters (temperature, top_p, presence_penalty, reasoning_effort, etc.)
- Jury-based title evaluation (multiple judges voting)

Example: A-B testing thinking enabled vs disabled in Qwen 3.5 by adding two experiment blocks with different `extra_body` settings.

**Troubleshooting Phoenix dataset cache:** If eval datasets have stale file paths after rebuilding, clear the Phoenix volume:
```bash
docker compose down -v phoenix
```
The next eval run will recreate it with fresh data.

## Code Organization

- `ai/src/paperless_ai/search/` — Indexing and retrieval
  - `webhook.py` — FastAPI webhook listener + `/search` endpoint (hybrid retrieval)
  - `retriever.py` — Core retrieval functions (dense, keyword, RRF, LLM rerank)
  - `embedder.py` — Dense embedding (LocalLazySearchEmbedder)
  - `qdrant_store.py` — Qdrant vector store
  - `queue.py` — Redis task queues
- `ai/src/paperless_ai/core/` — Core services
  - `paperless.py` — Paperless-ngx REST API client
  - `config.py` — Configuration from environment
- `ai/src/paperless_ai/agents/` — LLM pipelines
  - `smart_graph_agent.py` — OCR + metadata extraction (via LangGraph)
- `ai/src/paperless_ai/eval/` — Evaluation and metrics
- `ai/tests/` — Test suite
  - `conftest.py` — Session-scoped fixtures (Paperless token, clients, Redis state)
  - `test_search.py` — Unit + integration tests for embedder, retriever, webhook
  - `test_phase_b_pipeline.py` — Full OCR + metadata + embedding pipeline tests
