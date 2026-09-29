# Complete-PDF OCR services

Paperless AI sends the original base64 PDF to `POST /layout-parsing`. Choose
Paddle directly or the optional vision adapter. Metadata extraction remains a
separate stage. The endpoint selects a service; it never starts one or falls
back to another.

## Direct Paddle

Run the full Paddle parsing service on the workstation; the
[workstation implementation prompt](../paddle.md) describes that deployment.
Set the parsing service base URL, without `/v1`:

```env
INFERENCE_OCR_ENDPOINT=http://workstation:8080
INFERENCE_OCR_TIMEOUT=600
OCR_CONCURRENCY=1
```

Paddle selects its own recognition and layout models. No OCR model or
backend selector is configured in Paperless AI.

The service must expose `GET /metadata` with nonempty `pipeline`, `model`, and
`layout_model` strings identifying the running pipeline and models. Paperless AI
fetches this once before each OCR batch or job and records that identity with
the resulting documents. Keep `POST /layout-parsing`'s Paddle response
unmodified; it does not need a `result.provenance` extension. The deployed
Paddle service serves metadata with `Cache-Control: no-store`.

## Optional vision adapter

The independent `vision-ocr/` package translates the same PDF contract into
OpenAI-compatible vision requests. It needs no database, Redis, or GPU. It
processes one document at a time and renders pages sequentially, returning
page text blocks and Markdown, including empty page entries. It produces no
bounding boxes, layout detections, or image exports.

```env
INFERENCE_OCR_ENDPOINT=http://vision-ocr:8000
INFERENCE_OCR_TIMEOUT=600
OCR_CONCURRENCY=1
VISION_OCR_ENDPOINT=https://openrouter.ai/api/v1
VISION_OCR_MODEL=google/gemini-3.1-flash-lite
VISION_OCR_API_KEY=your-key
```

For vLLM, set `VISION_OCR_ENDPOINT=http://workstation:8100/v1` and its served
model name. The API key is optional for unauthenticated endpoints. Cloud
gateways must expose the OpenAI-compatible API; native Gemini and Anthropic
APIs are outside this adapter's scope.

```bash
docker compose up -d --build vision-ocr
docker compose stop vision-ocr
```

The `vision-ocr` Compose profile excludes it from ordinary startup. Explicitly
naming the service enables it, as described in the
[Docker profile documentation](https://docs.docker.com/compose/how-tos/profiles/).
It has `restart: "no"`, no published port, no Traefik route, and no dependencies
from normal services. It joins the existing internal and outbound networks.
Required adapter settings are validated only when the adapter starts.

| Adapter variable | Purpose / default |
|---|---|
| `VISION_OCR_ENDPOINT` | Required OpenAI-compatible base URL, including `/v1` where applicable |
| `VISION_OCR_MODEL` | Required served model ID |
| `VISION_OCR_API_KEY` | Optional upstream credential, separate from metadata/chat |
| `VISION_OCR_PROMPT` | Override the bundled transcription prompt |
| `VISION_OCR_MAX_IMAGE_DIMENSION` | Optional longest rendered edge in pixels; default rendering is 300 DPI |
| `VISION_OCR_MAX_TOKENS` | Completion limit, default `4096` |
| `VISION_OCR_TEMPERATURE` | Optional generation temperature |
| `VISION_OCR_REASONING_EFFORT` | Reasoning effort, default `minimal` |
| `VISION_OCR_EXTRA_KWARGS` | JSON generation options supported by the inference client |
| `VISION_OCR_TIMEOUT` | Whole document request timeout, default `600` seconds |

The adapter owns one inference client per lifespan and closes it at shutdown.
`GET /health` checks upstream `/models` with a short timeout and makes no
inference request. Invalid PDFs and unsupported request modes are rejected.
Upstream errors, timeouts, or explicitly truncated completions fail the whole
request. `GET /metadata` reports the configured model and uses
`layout_model="none"` because the adapter has no layout model. Retries remain in
Paperless AI's stage queue.

## Stored output and operation

| Field | Contents |
|---|---|
| `content` | Ordered Markdown, tables, equations, captions and text with image references removed |
| `ai_ocr_output` | JSON with document information, ordered page results, provenance, source SHA-256 and timestamp |
| `ai_result` | Metadata-stage output, preserved by OCR |

The parser validates complete page coverage. Before OCR begins, Paperless AI
fetches `GET /metadata` once per batch or job and requires nonempty `pipeline`,
`model`, and `layout_model` strings. It parses the standard, unmodified
`/layout-parsing` response and stores the service identity as provenance in
`ai_ocr_output`. If metadata is unavailable or invalid, processing fails before
any document writes. The public agent result reports
`ocr_method="layout-parsing"`.

Content, OCR output, and stage tags are persisted together after refreshing
tags and custom fields. Metadata preserves OCR output. Failed or incomplete
requests preserve existing document content and use the existing retry/failed
queue. An entirely empty transcript is rejected by the Paperless parser.
Readiness requires a successful `/health`; unavailable services leave work
queued. Run only one active processing worker process.

Pause new processing during backend upgrades, let active work finish, then
replace the service; an OCR batch must not span two deployments.

Recreate `ai` after changing its endpoint. Metadata/chat settings and credentials
stay in their existing services. Switching services affects future OCR work;
it does not migrate stored documents or reprocess the archive. To switch back
to a previous OpenAI-compatible `/v1` vision server, start `vision-ocr` against
that server, set `INFERENCE_OCR_ENDPOINT=http://vision-ocr:8000`, and recreate
`ai`. A raw `/v1` URL is not a `/layout-parsing` service. Existing stored results
need no migration, and changing URLs does not undo document writes.

Evaluations inherit the OCR endpoint and timeout, with per-experiment overrides
in `experiments.yaml`; their deadline adds 300 seconds for metadata. They use
the same complete-PDF interface and have no vision-only cache.

## Verification

Run adapter tests with `cd vision-ocr && uv run pytest tests/`. The cross-package
contract test also needs the AI dependencies: from `ai/`, run
`PYTHONPATH=../vision-ocr/src uv run pytest ../vision-ocr/tests/` after syncing
its test/evaluation extras. The Docker integration suite uses `./run_tests.sh`
from the repository root. These tests mock inference and do not reprocess
stored documents.
