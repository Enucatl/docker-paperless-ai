# docker-paperless-ai

AI batch OCR and metadata extraction for [paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) — no source patches required.

Documents are ingested normally via Tesseract, auto-tagged `ai-review-pending` by a paperless Workflow, then re-processed by this service: each page is re-OCRd with a vision LLM and title/date/correspondent are extracted with a text LLM. Everything is updated via the paperless REST API.

## Privacy notice

> **When using cloud models (the default), the following data is sent to the configured third-party API:**
>
> - **Page images** — every page of every processed document is sent to the OCR model.
> - **Document text** — the extracted text (or the first 6000 characters) is sent to the metadata model.
>
> Use `OCR_MODEL=ollama/...` or `OCR_MODEL=openai/...` with local servers for fully on-premises processing.

## How it works

```
New document arrives → Tesseract OCR (paperless default)
                     → tagged "ai-review-pending" (via Workflow)

AI worker runs → polls for tagged documents
              → downloads original PDF
              → OCRs each page with vision LLM
              → extracts metadata with text LLM
              → PATCHes document via REST API
              → removes tag "ai-review-pending"
              → writes processing note to document
```

The worker runs permanently alongside paperless and idles when no documents are pending or when model servers are unreachable. Documents queue up safely while the GPU workstation is off.

## Repo layout

```
docker-paperless-ai/
├── ai/
│   ├── cli.py                      # Entry point (--once, --watch, --eval, --dry-run, …)
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── prompt.txt                  # OCR instruction prompt (edit without rebuild)
│   ├── metadata_prompt.txt         # Metadata extraction prompt (edit without rebuild)
│   ├── agents/
│   │   ├── smart_graph_agent.py    # LangGraph-based vision OCR + metadata agent
│   │   └── base.py                 # AgentResult / DocumentMetadata types
│   ├── core/
│   │   ├── config.py               # AgentConfig — all settings from env vars
│   │   ├── paperless.py            # Paperless REST API client
│   │   ├── runner.py               # Redis-driven processing loop
│   │   └── telemetry.py            # OpenTelemetry → Arize Phoenix
│   ├── search/
│   │   ├── queue.py                # Redis Set queue (SADD/SMEMBERS/SREM, DB 1)
│   │   ├── webhook.py              # FastAPI listener — enqueues doc IDs from Paperless
│   │   ├── chunker.py              # Overlapping character-based text chunker
│   │   ├── embedder.py             # Infinity bge-m3 client (dense + sparse vectors)
│   │   └── qdrant_store.py         # Qdrant collection management
│   ├── eval/
│   │   ├── golden_dataset.json     # Ground truth for 50 IDL documents
│   │   ├── experiments.yaml        # Experiment configurations to compare
│   │   ├── run_evals.py            # Evaluation runner (called by --eval)
│   │   ├── metrics.py              # Scoring functions (fuzzy match, date distance, …)
│   │   ├── review_ground_truth.py  # Interactive annotation script
│   │   └── assign_splits.py        # One-time train/validation split assignment
│   └── tests/
│       ├── conftest.py             # Shared fixtures (queue, embedder mock, Qdrant, …)
│       ├── test_e2e_pipeline.py    # E2E: Redis queue → OCR → metadata → Paperless PATCH
│       ├── test_webhook.py         # Webhook listener + Paperless workflow integration
│       ├── test_metrics.py         # Unit tests for scoring functions
│       └── test_evaluator.py       # Unit tests for evaluation runner
├── docker-compose.yml              # Full server stack
├── docker-compose.override.yml     # Local volumes and secrets
├── docker-compose.test.yml         # Ephemeral E2E test override
├── run_tests.sh                    # One-command E2E test runner
└── .env.example                    # All environment variables documented
```

## Setup

### 1. Configure environment

```bash
cp .env.example .env
```

Set at minimum:

```env
PAPERLESS_SECRET_KEY=   # openssl rand -hex 32
DOCKER_DOMAIN=          # your domain
PAPERLESS_TOKEN=        # from paperless UI: Settings → API Tokens
GOOGLE_API_KEY=         # if using Gemini (default)
```

### 2. Create a paperless Workflow

In the paperless UI (Settings → Workflows):

- **Trigger:** Document Added
- **Action:** Assignment → add tag `ai-review-pending`

The tag is created automatically on first run if it doesn't exist.

### 3. Start the stack

```bash
docker compose up -d
```

This starts Redis, PostgreSQL, paperless-ngx, Gotenberg, Tika, and the AI worker.

### One-shot run

Process all pending documents and exit (useful for ad-hoc or scheduled runs):

```bash
docker compose run --rm ai --once
```

### Dry run

Preview actions without modifying any documents:

```bash
docker compose run --rm ai --once --dry-run
```

## Switching models

Edit `OCR_MODEL` (and optionally `METADATA_MODEL`) in `.env`, then restart:

```bash
docker compose restart ai
```

```env
# Gemini (default)
OCR_MODEL=gemini/gemini-2.5-flash

# Claude
OCR_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_API_KEY=your-key

# OpenAI
OCR_MODEL=gpt-4o
OPENAI_API_KEY=your-key

# Use a smarter model for metadata (called once per doc, not per page)
METADATA_MODEL=gemini/gemini-2.5-pro
```

## Local / self-hosted models

The AI worker connects to any OpenAI-compatible API via LiteLLM.

**Ollama** (easiest):

```env
OCR_MODEL=ollama/llava-llama3
METADATA_MODEL=ollama/llama3.2
OCR_API_BASE=http://workstation:11434
METADATA_API_BASE=http://workstation:11434
```

**vLLM** (recommended for Nanonets-OCR2-3B):

```env
OCR_MODEL=openai/nanonets/Nanonets-OCR2-3B
METADATA_MODEL=openai/meta-llama/Llama-3.2-3B-Instruct
OCR_API_BASE=http://workstation:8100/v1
METADATA_API_BASE=http://workstation:8101/v1
```

`OCR_API_BASE` and `METADATA_API_BASE` are independent — OCR and metadata can run on different servers or ports.

For running the model endpoints themselves, see [Enucatl/vllm](https://github.com/Enucatl/vllm).

## Docker secrets

API keys passed as plain env vars are visible in `docker inspect`. Use Docker secrets instead:

```yaml
# docker-compose.yml additions:
secrets:
  google_api_key:
    file: ./secrets/google_api_key.txt

services:
  ai:
    secrets:
      - google_api_key
    environment:
      - GOOGLE_API_KEY_FILE=/run/secrets/google_api_key
```

Supported `_FILE` variants: `GOOGLE_API_KEY_FILE`, `ANTHROPIC_API_KEY_FILE`, `OPENAI_API_KEY_FILE`, `PAPERLESS_TOKEN_FILE`.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `PAPERLESS_URL` | `http://webserver:8000` | Paperless base URL (internal Docker network) |
| `PAPERLESS_TOKEN` | *(required)* | API authentication token |
| `OCR_MODEL` | `gemini/gemini-2.5-flash` | LiteLLM vision model string for OCR |
| `METADATA_MODEL` | *(uses OCR_MODEL)* | LiteLLM text model for metadata extraction |
| `OCR_API_BASE` | *(none)* | Base URL for local OCR server |
| `METADATA_API_BASE` | *(none)* | Base URL for local metadata server |
| `GOOGLE_API_KEY` | *(none)* | For Gemini models |
| `ANTHROPIC_API_KEY` | *(none)* | For Claude models |
| `OPENAI_API_KEY` | *(none)* | For OpenAI / vLLM models |
| `POLL_INTERVAL` | `300` | Seconds between polls in watch mode |
| `TAG_PENDING` | `ai-review-pending` | Tag for documents awaiting processing |
| `OCR_REASONING_EFFORT` | `minimal` | LiteLLM `reasoning_effort` parameter (set empty to disable) |
| `DRY_RUN` | `false` | Log actions without modifying documents |

## Customising prompts

Edit `ai/prompt.txt` and `ai/metadata_prompt.txt` — they are mounted into the container as read-only volumes, so no rebuild is needed:

```bash
# Edit, then restart
docker compose restart ai
```

## Finding processed documents

On first run the worker creates a custom field **`ai_processed`** (type: Date) and sets it to the processing date on every document it finishes. The field does not appear as a tag — it shows in the document detail panel and is filterable in the search bar (`ai_processed is set` / `ai_processed is not set`).

## Reprocessing a document

Re-add the `ai-review-pending` tag and the worker will pick it up on the next poll. The service re-downloads the original and reprocesses, overwriting the previous content, title, and date.

To revert to Tesseract permanently, trigger a reprocess from the paperless UI (More → Reprocess document).

---

## Testing and evaluation

### E2E test suite

The test suite spins up a fully ephemeral stack (Paperless-ngx, Redis, Qdrant,
webhook-listener) in Docker, runs pytest inside the AI container, then tears
everything down — including all volumes.  No persistent state is left behind
even if the run is interrupted.

```bash
./run_tests.sh
```

On a warm Docker cache this takes roughly 2–3 minutes (dominated by Paperless
Django migrations and document indexing).  A fresh pull adds image download time.

#### What is tested

| Test file | What it covers |
|---|---|
| `test_e2e_pipeline.py` | Redis queue → OCR → metadata extraction → Paperless PATCH → queue drained |
| `test_e2e_pipeline.py` | Dry-run mode: Paperless unchanged, Redis queue intact |
| `test_e2e_pipeline.py` | Embedding pipeline: mock vectors upserted into real Qdrant with correct payload |
| `test_webhook.py` | Listener enqueues from `doc_url` field (Paperless `{{doc_url}}` placeholder) |
| `test_webhook.py` | Listener enqueues from `document_id` / `id` fallback fields |
| `test_webhook.py` | Redis SADD deduplication (same ID posted twice → one queue entry) |
| `test_webhook.py` | Graceful handling: missing ID → 202, non-JSON body → 400 |
| `test_webhook.py` | `/health` endpoint reflects live pending count |
| `test_webhook.py` | **Full Paperless integration**: workflow created via API → document uploaded → Paperless fires `{{doc_url}}` webhook → doc ID lands in Redis |
| `test_metrics.py` | Scoring function unit tests (correspondent, date, title) |
| `test_evaluator.py` | Evaluation framework unit tests |

#### Infrastructure used in tests

| Service | Image | Role |
|---|---|---|
| `webserver` | paperless-ngx | Real Paperless instance (tmpfs DB, anon volumes) |
| `broker` | redis:8 | Redis on tmpfs — DB 0 for Paperless, DB 1 for AI queue |
| `qdrant` | qdrant/qdrant | Vector DB (anonymous volume) |
| `webhook-listener` | *(this repo)* | Receives Paperless webhook events |
| `db` | postgres:17 | Paperless DB on tmpfs |

The Infinity embedding server is **not** available in the test environment (GPU
not present in CI).  The `mock_embedder` fixture provides deterministic 1024-d
fake vectors directly to `run_batch()` so the embedding code path is still
exercised end-to-end against real Qdrant.

#### Skip the build step (faster re-runs)

```bash
./run_tests.sh --no-build
```

#### Run a specific test file or test

```bash
# Run only the webhook tests
docker compose -f docker-compose.yml -f docker-compose.test.yml \
  run --rm ai bash -c \
  "uv pip install --system -e .[test] && pytest -v tests/test_webhook.py"

# Run a single test by name
docker compose -f docker-compose.yml -f docker-compose.test.yml \
  run --rm ai bash -c \
  "uv pip install --system -e .[test] && \
   pytest -v -k test_paperless_fires_webhook_on_document_added"
```

> These commands assume the infrastructure services (`webserver`, `qdrant`,
> `webhook-listener`, etc.) are already running.  Start them first with:
> ```bash
> docker compose -f docker-compose.yml -f docker-compose.test.yml \
>   up -d db broker gotenberg tika webserver qdrant webhook-listener
> ```

### Evaluation framework

The `ai/eval/` directory contains a golden dataset of 50 scanned documents (from the [IDL dataset](https://huggingface.co/datasets/aharley/rvl-cdip)) with ground-truth title, correspondent, and date annotations.

#### Ground truth annotation

Before running evaluations for the first time, annotate the golden dataset with the interactive review script. It runs the full agent pipeline on each document and prompts you to confirm or correct the proposed values:

```bash
docker compose run --rm --entrypoint python ai eval/review_ground_truth.py
```

For each document the script shows the OCR transcript and proposed title, correspondent, and date. Type `y` to accept, `n` to mark as genuinely null, `s` to skip, or enter a custom value.

Progress is saved after each document, so you can interrupt and resume at any time.

#### Train / validation split

After annotation, assign the train/validation split (one-time, deterministic):

```bash
docker compose run --rm --entrypoint python ai eval/assign_splits.py
```

This writes a `"split": "test" | "validation"` field to each entry in `golden_dataset.json`. Ten representative documents are held out as a validation set for prompt tuning and hyperparameter search; the remaining ~40 are the test set.

#### Running evaluations

Evaluations run all experiments defined in `ai/eval/experiments.yaml` and log results to Arize Phoenix (start it first with `docker compose up -d phoenix`).

```bash
# Smoke test — single tagged document, verifies the pipeline works end-to-end
docker compose run --rm ai --eval --split code-test

# Run against the test set (default)
docker compose run --rm ai --eval

# Run against the held-out validation set
docker compose run --rm ai --eval --split validation

# Run against all documents
docker compose run --rm ai --eval --split all
```

Each `--split` value maps to a separate named dataset in Phoenix (`paperless-golden-test`, `paperless-golden-validation`, `paperless-golden-code-test`, …), so experiments from different splits are never mixed in the comparison view.

The `code-test` split contains a single entry tagged `"tags": ["code-test"]` in `golden_dataset.json`. It is not filtered by the `"split"` field — any entry can carry the tag regardless of its train/validation assignment. To add more entries to the smoke test, add `"tags": ["code-test"]` to their entry.

#### Metrics

Each evaluation run reports per-experiment:

| Metric | Description |
|---|---|
| `correspondent_exact_accuracy` | Exact match after case-normalisation and suffix removal (Inc., AG, …) |
| `correspondent_fuzzy_mean` | Token-sort fuzzy score — robust to word reordering |
| `date_exact_accuracy` | Exact ISO date match |
| `date_partial_mean` | Partial credit: linear decay from 1.0 (exact) to 0.0 (≥ 1 year off) |
| `null_precision` / `null_recall` | Precision/recall for documents where no correspondent exists |
| `title_contains_rate` | Fraction where actual title contains the expected keyword |

A comparison table is printed at the end of each run:

```
=== Experiment Comparison ===
  baseline-flash:  corr_exact=65.0%  corr_fuzzy=0.82  date_exact=72.0%  date_partial=0.89
  creative-flash:  corr_exact=60.0%  corr_fuzzy=0.79  date_exact=68.0%  date_partial=0.85
```

#### Adding experiments

Edit `ai/eval/experiments.yaml` — no rebuild required. Any `AgentConfig` field can be overridden per experiment:

```yaml
experiments:
  - name: "baseline-flash"
    ocr_model: "gemini/gemini-2.5-flash"
    metadata_model: "gemini/gemini-2.5-flash"
    temperature: 0.0

  - name: "local-nuextract"
    ocr_model: "openai/Nanonets-OCR2-3B"
    ocr_api_base: "http://workstation:8100/v1"
    metadata_model: "openai/numind/NuExtract-2.0-4B"
    metadata_api_base: "http://workstation:8101/v1"
    temperature: 0.0
```
