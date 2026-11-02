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

## Searchable archival PDFs

The OCR worker disables document unwarping (`useDocUnwarping=false`) and does
not request corrected page images. OCR preprocessing is used for recognition,
not as the visual source of the archive.

The AI service produces PDF/A-2u using Ghostscript and veraPDF, preserving
native text, vector content and embedded image resolution. Pages with no existing
extractable text receive an invisible OCR text layer; pages with existing text
are left alone to avoid duplicate selection and search results. This is a
presence check, not a text-density or image-count threshold. A page combining
native text and scanned regions keeps its existing text only.

Before conversion, simple fonts explicitly declaring `WinAnsiEncoding` receive
missing Unicode maps derived from that encoding. Existing Unicode maps are
preserved. For undeclared symbolic TrueType fonts, embedded Unicode tables and
real glyph names are preferred. If they are inconclusive, consistent valid
Unicode interpretations from the source PDF reader are made explicit. Other
unsupported fonts retain the converter's normal interpretation. This may retain
imperfect character guesses; Unicode conformance does not guarantee semantic
accuracy.

Only new OCR fonts are subset, preserving existing text mappings. After all
edits, the archive declares PDF/A-2u and must pass the pinned veraPDF validator's
2u profile before publication. Failed validation leaves the existing archive
unchanged. There is no OCR-based font matching or non-PDF/A fallback.

The overlay uses Paddle block bounding boxes, so selection within paragraphs
and tables is approximate. Original page images are never replaced with OCR
preprocessing images. Orientation corrections are mapped back onto the original
page; unwarped OCR is rejected for overlays because its nonlinear geometry
cannot be mapped back this way. The full OCR transcript remains searchable in
Paperless.

It sends the archive and transcript together to `POST /api/documents/{id}/archive/`,
with the original's SHA-256 as `source_checksum`. Paperless preserves the original
and rejects stale results. Stage tags and stored OCR provenance are updated only
after the archive upload succeeds.
The vision adapter (`layout_model=none`) continues its text-only workflow.

On Atlas, prohibit Paperless's local OCR, classifier, and built-in AI via
`PAPERLESS_LOCAL_OCR_ENABLED=false`, `PAPERLESS_CLASSIFIER_ENABLED=false`, and
`PAPERLESS_AI_ALLOWED=false`. These restrictions leave this external archival
workflow available. Deploy a Paperless image supporting the archive endpoint
and deployment restrictions before updating the external worker.
