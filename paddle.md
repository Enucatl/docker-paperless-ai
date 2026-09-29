# Workstation implementation prompt: PaddleOCR-VL-1.6

Work in `/export/complex/vllm`. Deploy the official **full PaddleOCR-VL-1.6
document parsing pipeline**, with **PP-DocLayoutV3** layout detection and
**vLLM** recognition, on the **RTX 5090 32 GB**. Paperless AI already has an
HTTP client for the contract below. This task implements the workstation
service; Paperless cutover is a separate, gated operation.

## Inspect, deploy, retain rollback

Read this repository's instructions, deployment conventions, current Compose
files, model caches, endpoints and GPU allocations before editing. Reuse those
patterns and select a free parsing-service port reachable from the Paperless
Docker host. Keep this API separate from existing OpenAI model endpoints.

Use the current official [Blackwell deployment
guide](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html)
and its linked Compose configuration. The guide specifies a driver supporting
CUDA 12.9 or newer and `nvidia-gpu-sm120` images for both the parsing service and
vLLM service. Its published hardware verification is RTX 5070, so test the
actual 5090 deployment. Record tested image **digests**, installed versions and
both model **revisions**; do not report floating tags as reproducible pins.

Select the 1.6 pipeline explicitly; ensure the layout model is PP-DocLayoutV3.
The [official vLLM recipe](https://recipes.vllm.ai/PaddlePaddle/PaddleOCR-VL-1.6)
documents `pipeline_version="v1.6"` and the need to match the recognition API
model name to the name served by vLLM. Follow the Blackwell guide for runtime
images and CUDA compatibility instead of copying a generic GPU installation.
The recognition server alone is only one stage; Paperless needs the full PDF
parsing API that renders pages, detects layout and produces Markdown/JSON.

Retain the existing Nanonets deployment configuration and model cache for
rollback. Stop it if necessary to free GPU memory; simultaneous GPU residency
is unnecessary. Keep metadata/chat endpoints operational according to this
repository's existing allocation conventions. Keep GPU dependencies out of
the Paperless AI image.

## Exact Paperless client contract

The client configures
`INFERENCE_OCR_ENDPOINT=http://<reachable-host>:<parsing-port>` **without `/v1`**,
`INFERENCE_OCR_TIMEOUT=600` seconds, and initially `OCR_CONCURRENCY=1`.
The URL selects the parser; there is no backend selector or OCR model setting
in Paperless AI. The parsing service runs independently on the workstation.

`GET /health` must return a successful HTTP response only when the complete
pipeline is ready, including its recognition dependency. Return a failing
status while loading or unavailable; a 404 or a merely listening socket is not
readiness.

`GET /metadata` must return nonempty `pipeline`, `model`, and `layout_model`
strings for the deployed pipeline and both models, for example:

```json
{"pipeline": "PaddleOCR-VL-1.6", "model": "PaddlePaddle/PaddleOCR-VL-1.6", "layout_model": "PP-DocLayoutV3"}
```

Paperless AI fetches this once per OCR batch or job before any document writes.
An unavailable or invalid response fails processing. Pause new Paperless AI
processing during backend upgrades, let active work finish, then replace the
service so one batch cannot span deployments.

`POST /layout-parsing` accepts JSON with the **entire original PDF**, not
individual images or a document URL:

```json
{
  "file": "<base64-encoded complete PDF bytes>",
  "fileType": 0,
  "useLayoutDetection": true,
  "markdownIgnoreLabels": [],
  "returnMarkdownImages": false,
  "visualize": false,
  "restructurePages": false
}
```

Return the official synchronous parsing response shape. This illustrative
one-page shape shows the required fields; preserve the actual structured
results produced by Paddle. Do not add provenance to this response; Paperless
AI records the identity from `/metadata` separately:

```json
{
  "logId": "request-id",
  "errorCode": 0,
  "errorMsg": "Success",
  "result": {
    "dataInfo": {
      "type": "pdf",
      "numPages": 1,
      "pages": [{"width": 595, "height": 842}]
    },
    "layoutParsingResults": [
      {
        "prunedResult": {"parsing_res_list": []},
        "markdown": {"text": "Recognized page text", "images": null}
      }
    ]
  }
}
```

Each source page must have one valid result, in source order, including blank
pages whose Markdown may be empty. Return all Markdown labels, including
headers/footers and captions, plus tables, equations and Unicode text. Keep
page results separate: no cross-page restructuring or concatenation. The
official API removes `page_index` from `prunedResult`; Paperless assigns
explicit zero-based page indices using response order and verifies the count
against the source PDF. Do not drop a blank page or silently return a partial
document on errors. The client rejects wholly empty transcripts, malformed
results, count mismatches and HTTP/API errors before any content write.

Honor `returnMarkdownImages=false`: do not encode, upload or return Markdown
image payloads. Honor `visualize=false` and disable visualization by default.
Do not request or add binary exports. Image fields may be omitted or null.
The client strips image references while retaining captions and recognized
text, and persists only sanitized document information, page `prunedResult`
and Markdown with provenance. It does not store images.

The [official parsing API and serving
configuration](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/PaddleOCR-VL.html#43-client-side-invocation)
describe these request/response fields. Set the full-service pipeline config:

```yaml
Serving:
  visualize: false
  extra:
    max_num_input_imgs: null
```

`max_num_input_imgs: null` is mandatory: complete transcripts must never be
truncated by server-side page caps. Configure proxy/body limits for base64
PDFs and large structured responses. Request, proxy and recognition timeouts
must accommodate the client's 600-second default; if tested long documents
need more, report the matching timeout setting for both repositories.

## Operational requirements and verification

Persist model caches using existing repository conventions. Start with one
concurrent document and measure total GPU memory for layout plus recognition;
leave room for the actual model workload and any retained services. Use
readiness checks and dependency startup ordering. Pin the tested artifacts
after the service works on the 5090.

Provide runnable **startup, health, smoke-test, shutdown and rollback commands**
using the actual service names and ports you implement. The smoke test must
send the exact complete-PDF JSON body above, wait for completion, check
`errorCode == 0`, and compare source PDF page count with returned pages.

Verify a small fixed sample covering multipage PDFs, tables, Unicode, an
individual blank page and a PDF **longer than 40 pages**. Check first, middle
and last page text/order; valid JSON; every page represented; table/caption/
equation retention; no image or binary-export payloads. Exercise an invalid
PDF/error response and unhealthy recognition dependency. Record elapsed time
and observed GPU memory, including the long PDF. No comparative experiment,
quality scoring, image storage or archive-wide backfill.

Report the reachable parsing **base URL**, `/layout-parsing`, `/health`, and
`/metadata` URLs from the Paperless Docker network; exact startup, shutdown,
and rollback commands; GPU memory; tested image digests and model revisions;
sample page counts and outcomes; any required timeout changes. Do not claim
completion solely from the vLLM recognition server starting.

Paperless then snapshots a small fixed pilot's content, metadata, tags and
custom fields, processes only that pilot, verifies complete text and field
ownership, and enables Paddle for incoming documents after checks pass. Keep
the single-active-worker deployment assumption: stop the Paperless `ai`
service before a processing CLI instance. Switching to another
`/layout-parsing` service, or routing the previous OpenAI-compatible OCR server
through `vision-ocr`, affects future processing; previous document writes
require the saved pilot snapshots.
