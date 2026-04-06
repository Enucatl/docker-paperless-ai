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
uv run pytest tests/ -k "not test_webhook and not test_phase_b_pipeline and not test_search"
```

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

**Important:** Full E2E tests require Docker infrastructure. Use `uv run pytest` for unit-level testing without Docker.

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
