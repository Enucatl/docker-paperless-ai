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
│   ├── batch.py              # AI worker service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── prompt.txt            # OCR instruction prompt (edit without rebuild)
│   └── metadata_prompt.txt   # Metadata extraction prompt (edit without rebuild)
├── docker-compose.yml        # Full server stack (paperless + AI worker)
└── .env.example              # All environment variables documented
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

## Reprocessing a document

Re-add the `ai-review-pending` tag and the worker will pick it up on the next poll. The service re-downloads the original and reprocesses, overwriting the previous content, title, and date.

To revert to Tesseract permanently, trigger a reprocess from the paperless UI (More → Reprocess document).
