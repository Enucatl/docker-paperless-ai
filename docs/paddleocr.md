# PaddleOCR-VL-1.6 operations

The `paddleocr` backend sends the complete PDF to a GPU workstation's full
document parsing service. Deploy that service using the [workstation
prompt](../paddle.md) in `/export/complex/vllm`. GPU libraries and model caches
stay on the workstation; the AI image remains a lightweight HTTP client.

## Configuration and storage

Keep `INFERENCE_OCR_BACKEND=vision` until the workstation service and the fixed
pilot below pass. After cutover, use:

```env
INFERENCE_OCR_BACKEND=paddleocr
INFERENCE_OCR_ENDPOINT=http://workstation:8080
INFERENCE_PADDLE_TIMEOUT=600
OCR_CONCURRENCY=1
```

Replace the example host/port with the workstation agent's tested parsing URL.
This URL has no `/v1` suffix. `INFERENCE_OCR_MODEL` and vision generation options
apply only to the legacy backend; Paddle's model is selected by its service.
Metadata and chat retain their existing model endpoints. Keep the old OCR
endpoint, model and deployment configuration for rollback.

| Paperless field | Owner and content |
|---|---|
| `content` | OCR: ordered Markdown, tables, equations, captions and text, without image references |
| `ai_ocr_output` | Paddle OCR: long-text JSON containing document information, each page's `prunedResult` and Markdown, zero-based indices, schema version, model/pipeline identity, source SHA-256 and timestamp |
| `ai_result` | Existing metadata model result |

Service and processing CLI startup create `ai_ocr_output`. The OCR worker
updates content, OCR output and stage tags in one PATCH after refreshing tags
and custom fields. Metadata preserves the OCR field; a successful legacy OCR
rerun removes it. Other custom fields remain intact.

Both OCR backends process every page. The Paddle client counts source PDF pages
without rendering and rejects missing/malformed results, API errors and wholly
empty transcripts. An individual blank page is valid. Failures retain existing
content/output and use the normal stage retry/failed queue. Paddle readiness
requires a successful `/health` response. Increase the positive request timeout
if the verified long-document run needs more than 600 seconds; proxy and server
timeouts must accommodate it too.

Only **one active processing worker process** may run. Same-document OCR and
metadata share a process-local lock; separate documents remain concurrent.
Stop `ai` before launching a processing CLI instance. The lock does not
coordinate different processes or external Paperless edits.

## Fixed pilot and cutover

The endpoint is an external deployment prerequisite. Do not enable Paddle for
incoming documents before the workstation agent reports tested API/health URLs,
complete page coverage (including a PDF longer than 40 pages), image suppression,
model revisions, image digests and measured GPU memory.

1. Choose a small fixed list of existing PDF IDs with a multipage document,
   table, Unicode and an individual blank page. Include a document longer than
   40 pages. Record the list once; keep failures in the pilot. Do not enqueue
   the rest of the archive.
2. Pause imports and external edits for the pilot, let existing ready and
   delayed OCR/metadata jobs drain using the current backend, then stop `ai`
   and `webhook-listener`. Confirm the queues are empty, including delayed
   retries. This prevents the one-shot run from touching unrelated documents.
3. Snapshot each pilot's complete API response before adding any stage tags.
   These files contain private document text; keep them local and protected.
   Set `PAPERLESS_URL` to a reachable Paperless API base, `PAPERLESS_TOKEN` to
   its token, and replace these example IDs with the fixed list:

   ```bash
   docker compose stop ai webhook-listener
   export PILOT_IDS="101 202 303"
   umask 077
   mkdir -p tmp/paddle-pilot
   printf '%s\n' "$PILOT_IDS" > tmp/paddle-pilot/ids.txt
   cp .env tmp/paddle-pilot/env.before
   for id in $PILOT_IDS; do
     curl --fail --silent --show-error \
       -H "Authorization: Token $PAPERLESS_TOKEN" \
       "$PAPERLESS_URL/api/documents/$id/" \
       -o "tmp/paddle-pilot/$id.before.json" || break
   done
   ```

   Verify every snapshot parses as JSON and contains the matching ID, content,
   metadata, tags and custom fields before continuing. Preserve original PDF
   SHA-256 values for the later provenance check as well.
4. Build the updated AI image, leaving the service stopped. Enqueue exactly
   the fixed pilot IDs using the existing queue helper:

   ```bash
   docker compose build ai
   docker compose run --rm --no-deps -T -e PILOT_IDS \
     --entrypoint python ai - <<'PY'
   import asyncio
   import os
   from paperless_ai.core.config import AgentConfig
   from paperless_common.queue import TaskQueues

   async def main():
       queues = TaskQueues(AgentConfig.from_env().redis_url)
       try:
           for doc_id in map(int, os.environ["PILOT_IDS"].split()):
               await queues.enqueue_ocr(doc_id)
       finally:
           await queues.close()

   asyncio.run(main())
   PY
   docker compose run --rm --no-deps \
     -e INFERENCE_OCR_BACKEND=paddleocr \
     -e INFERENCE_OCR_ENDPOINT=http://workstation:8080 \
     -e INFERENCE_PADDLE_TIMEOUT=600 -e OCR_CONCURRENCY=1 \
     -e MANAGE_PAPERLESS_WORKFLOWS=false \
     --entrypoint python ai cli.py --once
   ```

   Use the tested URL in the command. A failed run can schedule delayed retries;
   inspect outcomes and retry those fixed IDs through the normal mechanism
   while keeping other processing stopped. A zero shell exit code alone does
   not establish document success.
5. Fetch the same IDs as `*.after.json`. For every document, confirm OCR and
   metadata completed and no failed/retry entry remains; parse `ai_ocr_output`
   as JSON; compare source SHA-256 and every page index/count with the original
   PDF. Check there are no image/binary payloads or image references. Read the
   first, middle and last pages, table and Unicode text; search Paperless for
   distinctive recognized phrases. Confirm OCR preserved `ai_result` and other
   metadata fields before the metadata stage, metadata preserved the exact OCR
   JSON, and unrelated custom fields/tags survived both stages. Metadata-owned
   fields may change under the existing extraction rules. Record each pilot
   outcome and retain snapshots. This is an operational check, not a comparison
   experiment or quality-scoring project.
6. Once those checks pass, set the four Paddle variables above in `.env`, then
   run `docker compose up -d --force-recreate ai webhook-listener` and resume
   imports. New documents use Paddle. Leave historical documents untouched
   except for this pilot.

## Rollback

Stop the AI service before rollback, clear or remove pending pilot retries
through the existing queue API, and keep pilot edits/imports paused. On the
workstation stop Paddle and start the retained Nanonets service using the
commands supplied by its agent. Restore `INFERENCE_OCR_BACKEND=vision`, the
original OCR endpoint/model and concurrency values, then recreate `ai` when
ready. A configuration rollback only changes future work; **it does not undo
document writes**.

To undo pilot writes, restore the fields the pipeline can change from each
saved snapshot while the workers/listener remain stopped:

```bash
for id in $PILOT_IDS; do
  jq '{content,title,created,correspondent,document_type,storage_path,tags,custom_fields}' \
    "tmp/paddle-pilot/$id.before.json" > "tmp/paddle-pilot/$id.restore.json"
  curl --fail --silent --show-error -X PATCH \
    -H "Authorization: Token $PAPERLESS_TOKEN" -H 'Content-Type: application/json' \
    --data-binary "@tmp/paddle-pilot/$id.restore.json" \
    "$PAPERLESS_URL/api/documents/$id/" \
    -o "tmp/paddle-pilot/$id.restored.json" || break
done
```

Read back each document and compare the restored fields with its snapshot.
Remove any pilot queue entries generated by restoration before restarting the
listener and worker, then resume imports. Retain snapshots until restoration
is verified. Do not restore over later deliberate user edits without reviewing
those changes.
