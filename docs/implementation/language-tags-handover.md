# Language tags handover — 2026-09-27

## Language tag color follow-up

The user requested dark green language tags. Existing `language:de`,
`language:en` and `language:it` tags were updated through the API to `#166534`;
the other tags' colors and matching settings were verified unchanged. The
creation path now supplies the same color. Focused client/persistence tests:
19 passed in 0.12 seconds. Worker deployment completed successfully in run
`.codex-wake-run/d19a19397519.log`; the service is healthy and the deployed code
was checked for the dark-green creation default. No archive backfill was
requested by the color change.

## Pilot complete — awaiting user review

Run `.codex-wake-run/05a1b6812a9d.log` completed successfully. All ten eligible
documents (2486–2477) succeeded, with no skips/failures. Writes completed between
19:25:28 and 19:25:32 UTC on 2026-09-27. OCR hashes, document dates and addition
timestamps are unchanged. Every saved language tag matches its audit array;
new language tags have matching_algorithm=0. Language/text review found no
missing or spurious substantive languages (review used stored OCR, not PDFs).

- Italian: 2486, 2485, 2484, 2479, 2478.
- English: 2483, 2477.
- English and Italian: 2482 (Italian ticket instructions and English theatre rules).
- German: 2481, 2480.

All summaries, processing dates and audit records refreshed. Eight titles
changed. Document 2477's correspondent changed from YouTube to Matteo Abis;
the stored payment slip names the latter as payee. This is a metadata choice
for pilot review, not a language persistence failure.

Full snapshots and exact before/after changes are in ignored operational files
`tmp/language-pilot/before.json`, `after.json`, and `report.md`. The updated worker
is deployed and healthy. The test suite passed 222 tests in 20.68 seconds
(one opt-in live-model test skipped; infrastructure startup 31 seconds).
**Stop for user acceptance of the pilot. No archive backfill has been started.**

## Current selection (supersedes the original newest-ten rule)

The user explicitly changed the pilot to **the newest ten documents without
pending OCR**. The fresh snapshot at `tmp/language-pilot/before.json` selects
**2486, 2485, 2484, 2483, 2482, 2481, 2480, 2479, 2478, 2477**; it excludes
2488 and 2487. Selection checks OCR tags plus ready and delayed OCR queues while
paging in `-added,-id` order. All ten also had clear metadata queues at snapshot.
The original snapshot is preserved as `before-original-newest-ten.json`.
An isolated selection regression check passed in 0.09 seconds. The user did not
authorize holding or changing the existing OCR jobs; none were changed.
Proceed with the fixed eligible pilot and stop for review before archive backfill.

All ten eligible IDs have now been enqueued through `TaskQueues.enqueue_metadata`
after a full-detail snapshot was copied outside the container. The list API omits
duplicate-document details for some documents; snapshotting the detail API fixed
that false change detection before enqueue. `run-pilot.sh` now only waits and
saves results, so it will not enqueue the pilot again.

## Resume update on docker.home.arpa

The user requires the test suite to finish in seconds. This requirement is now
recorded in `AGENTS.md` and the testing reference. The upload fixture now handles
Paperless v10 lowercase statuses and `result_data.document_id`, polls every
100 ms, and fails with diagnostics after 10 seconds. The task UUID filter **is**
supported by `TasksViewSet.get_queryset`; the original filter suspicion below
was disproved. Six fixture regression cases pass in 0.14 seconds; the 81 focused
language tests pass in 0.25 seconds.

Test Redis persistence is disabled, readiness checks are faster, and webhook
polling uses 100 ms intervals and 10-second deadlines. The harness stops on its
first failure and reports slow tests and separate build/startup times.

Run `.codex-wake-run/cccd2128c281.log` reached **214 passed, 1 failed, 1 skipped
in 24.95 seconds** (build 4 seconds; initial infrastructure startup 32 seconds).
The remaining failure exposed a test listener bootstrap problem: its read-only
root prevented writing `/tmp/paperless-token`, silently disabling Paperless
integration. Both test service startup commands now capture the token in a
shell variable and use `bash -e`; the listener waits for healthy Paperless.
The subsequent run `.codex-wake-run/378c131bb433.log` passed: **222 passed,
1 skipped in 20.68 seconds**, plus 31 seconds of infrastructure startup. The
skip is the opt-in live-model contract test. Test containers were removed.
The test webserver also sets `CELERY_WORKER_MAX_TASKS_PER_CHILD=100` to avoid
Paperless's default child replacement after each upload; installed Celery CLI
parsing confirmed this native environment override before use.

Production deployment and the fixed ten-document pilot are the next step. The
deployment environment was compared to the live worker with no differing keys.
A fresh read-only review found no blocking language implementation or pilot
script issue. The existing worker polls every 300 seconds, so the real pilot
can take several minutes even though the mocked test suite finishes in seconds.

Worker deployment subsequently succeeded (healthy) in run
`.codex-wake-run/10ca0c7646a0.log`. Copying the operational script into the
read-only container failed before snapshot or enqueue. The pilot runner now
executes the script through `docker exec -i ... python - < pilot.py` and uses
the writable `/tmp` mount for snapshots. It resumes on the deployed worker
without rebuilding or recreating it.

Pilot preflight found existing ready OCR jobs for documents **2488 and 2487**;
the OCR server `http://complex.home.arpa:8100/v1` is unreachable (Loki worker
startup warning). No metadata was enqueued. User input is pending on temporarily
holding and restoring those two jobs versus waiting for their existing OCR.
The fixed newest-ten snapshot is now safely saved at
`tmp/language-pilot/before.json`: IDs **2488 through 2479**, corpus count 2387,
including content hashes, metadata, resources and ready/delayed queue state.
Do not resnapshot or substitute documents. The enqueue mode retains the pending
queue guard. Docker archive copying cannot see the `/tmp` mount; transfer its
JSON snapshots using `docker exec ... cat ... > local.json` instead.

## Resume location and stopping point

The user requested this handover and termination of activities in the original
Codex session so work can resume directly on **docker.home.arpa**.

- Live Compose project: `/opt/docker/paperless-ai` on `docker.home.arpa`.
- The original workspace was `/export/docker/paperless-ai`. Both remote paths
  expose the same edited source; matching Git status and source hashes were
  verified. Changes are already visible on the Docker host. Do not copy an old
  checkout over them.
- Branch: `main`, all changes uncommitted. No commit or push was requested.
- **Production has not been deployed, and no production documents have been
  snapshotted, enqueued, or changed by this task.**
- The full Docker integration harness remains failing. Do not call it passed.
- All three subagents completed. The last test command completed and removed
  its test containers. No wake-run process remained in the original environment
  when preparing this handover.

The Compose invocation over noninteractive SSH needed
`DOCKER_DOMAIN=docker.home.arpa`; otherwise `extra_hosts` failed validation.
The existing live AI container's router rule confirmed this domain. The host
also has `/opt/docker/.env`, where `DOCKER_DOMAIN=docker.${DOMAIN}` is defined.
Do not print secrets or replace the existing deployment environment.

## User intent and acceptance boundary

Extract all substantive document languages in the existing metadata AI call.
Store current classifications as reserved Paperless tags such as `language:de`
and `language:en`; record the extracted array or null in
`ai_result.ai_metadata.languages`. No custom language field, database migration,
or search prompt change.

After verification, deploy only the updated worker, then:

1. Select **exactly the 10 newest documents** using
   `ordering=-added,-id&page_size=10`.
2. Snapshot fixed IDs, added timestamps, metadata, tags, custom fields, and OCR
   content hashes before any enqueue.
3. Enqueue those IDs through `TaskQueues.enqueue_metadata` for the existing
   worker. Run metadata extraction from stored OCR text; do not initiate OCR.
4. Report each document's detected languages, actual metadata changes, and
   success/failure/skip. Check languages against text and verify unchanged OCR
   hashes. Empty-content documents remain in the pilot and are reported as
   skipped; do not substitute older documents.
5. Resolve language extraction/persistence defects found in the pilot.
6. **Stop for pilot review.** Only after user acceptance, snapshot and enqueue
   the remaining archive, excluding successful pilot documents, and report all
   outcomes.

Metadata-only reruns intentionally also refresh titles, dates, correspondents,
summaries, processing dates, and audit records under the existing rules.

## Implemented changes

Read `git diff` for the actual code; the following is the current intent.

- `ai/src/paperless_ai/agents/base.py`: optional public `languages` array.
- `ai/src/paperless_ai/agents/smart_graph_agent.py`: optional extracted languages;
  before-validator trims, lowercases, sorts, deduplicates, and discards entries
  other than two or three ASCII letters. Non-list/empty results become `None`.
  Instructions request ISO 639-1 codes where available, otherwise ISO 639-3,
  ignoring incidental names and isolated foreign words. Structured-output and
  NuExtract template/prompt/normal and fallback parsers carry languages, as does
  the graph's public result. No extra model request.
- `common/src/paperless_common/paperless.py`: `get_tag_id` has optional
  `matching_algorithm`; existing callers retain default behavior. New tag
  creation invalidates `_tags_cache` and reuses `_tag_id_cache`.
- `ai/src/paperless_ai/core/runner.py`: batch-local lock serializes language tag
  discovery/resolution. First confident result loads language IDs with fresh
  tags; new codes resolve through the existing ID cache and use
  `matching_algorithm=0`. Confident results replace only `language:` tags;
  unknown results preserve them. Unrelated tags survive, and the metadata stage
  tag is removed. Tags and metadata/audit are written in the same PATCH. Tag
  resolution failures enter existing retry handling before any document PATCH.
  Unused tags created before a subsequent resolution failure can remain, by
  design; corpus inventory filters these out.
- `docs/reference.md`: tag convention, fresh-count corpus inventory, document
  tag filters, and metadata-only reprocessing/pilot procedure.
- Tests: extraction and graph propagation, tag helper/cache behavior, new
  `ai/tests/test_language_tags.py` for concurrent reuse, unknown preservation,
  no partial PATCH on resolution failure, audit and unrelated fields, dry run.
  Extended metadata Docker integration test checks stale replacement,
  unrelated tags, audit array, unchanged OCR, bilingual filtering, fresh counts
  excluding unused language tags, and disabled automatic matching.

Existing matching settings on already-existing language tags are not modified;
the requested disabled setting applies to newly created tags.

## Verification already completed

- Common and AI Python formatting ran with `uv run ruff format .` (or
  `uv run --no-sync ruff format .` where the environment was already synced).
- `git diff --check` passed before the latest handover.
- Focused tests: **81 passed** with:

  ```bash
  cd ai
  uv run --no-sync python -m pytest \
    tests/test_extraction_strategies.py \
    tests/test_paperless_client.py \
    tests/test_language_tags.py -q
  ```

- The environment was synced with test/eval extras. In the original workspace
  the `pytest` executable had a stale `/opt/docker/...` shebang; using
  `python -m pytest` worked. This may naturally resolve when running on the
  Docker host at `/opt/docker/paperless-ai`.
- A separate code review found no blocking issue in extraction/persistence.

## Docker harness failures and fixes so far

Commands ran on `docker.home.arpa`, from `/opt/docker/paperless-ai`, with
`DOCKER_DOMAIN=docker.home.arpa ./run_tests.sh` (or `--no-build`). The harness
uses project `paperless-ai-test`; production containers were not changed.

Logs, all preserved in the shared checkout:

| Log | Outcome |
| --- | --- |
| `.codex-wake-run/5c688c43e201.log` | Built images; test Phoenix unhealthy because ephemeral PostgreSQL has no Phoenix role/database. No pytest. |
| `.codex-wake-run/191674816c87.log` | First SQLite override mistakenly included `python` despite the image's Python entrypoint. No pytest. Corrected. |
| `.codex-wake-run/fcedf06d5dd1.log` | 199 passed, 13 failed, 1 skipped, 4 errors; exposed stale integration fixtures. |
| `.codex-wake-run/9643fdd3266a.log` | **Latest:** 189 passed, 5 failed, 23 skipped, 1 error in **609.19 seconds**. Five upload waits each exhausted 120 seconds; Redis later rejected writes. |

Test-only repairs already made, uncommitted:

- `docker-compose.test.yml`: Phoenix runs using its existing Python entrypoint
  with command `["-m", "phoenix.server.main", "serve"]`, an overridden
  environment containing `PHOENIX_WORKING_DIR=/tmp/phoenix`, and no inherited
  volumes/secrets. This fixed startup; source inspection confirmed the image
  defaults to SQLite without PostgreSQL variables.
- Same file: test `ai` gets `POLL_INTERVAL=86400` so its initial empty startup
  batch is followed by a long sleep. Tests explicitly call batches or inspect
  queued state; the live background worker must not drain their queues.
- `ai/tests/conftest.py`: `_upload_document` now unwraps paginated task results
  (`tasks["results"]` when response is a dict). **This is not sufficient for the
  current Paperless API; see the latest findings below.**
- `ai/tests/test_webhook.py`: restores the missing callable `uploaded_document`
  fixture with cleanup; adds processing tags to synthetic webhook payloads;
  test workflows assign an OCR tag before sending their webhook (the listener
  correctly ignores untagged current documents); cleans up workflows created by
  the auto-managed-workflow test. Health requests use `webhook_session` as
  required by AGENTS instructions.

These are existing harness/fixture problems exposed by running the required
integration suite. Do not change production listener behavior to satisfy stale
tests.

## Latest concrete findings — investigate these next

The ten-minute run was not slow language extraction. The shared upload fixture
failed to recognize completed Paperless tasks, and then test Redis persistence
failed. No next test run has been started.

1. **A timed-out upload actually completed in under one second.** Loki showed
   task `74b8870a-f995-442e-899d-3fceea8065a1` received at
   `2026-09-27 20:51:12` Europe/Zurich and succeeded at `20:51:13`, with Celery
   result `{'document_id': 1}`. The fixture still waited 120 seconds. It currently
   requests `/api/tasks/?task_id=<uuid>`, takes the first result, expects status
   `SUCCESS`, and reads `related_document` or a number from string `result`.
   Inspect the actual current task API serializer, status values, and lookup
   route/filter before changing it again.

2. **Current Paperless source suggests `task_id` is no longer a supported list
   filter.** A read-only inspection of the live image's
   `/usr/src/paperless/src/documents/filters.py` showed
   `PaperlessTaskFilterSet.Meta.fields` containing `task_type`, `trigger_source`,
   `status`, `acknowledged`, `owner`, `name`, `result`, but no `task_id`.
   Therefore the fixture may be reading an unrelated first task. This is a
   strong lead, not a completed diagnosis. The attempted search for
   `PaperlessTaskSerializer` in `documents/serialisers.py` did not find that
   exact class name. Use the actual source/API rather than guessing the new
   schema.

3. **Test Redis cannot write `/data`.** Loki records
   `Failed opening the temp RDB file ... (in server root dir /data) for saving:
   Permission denied`. It later raises `MISCONF ... stop-writes-on-bgsave-error`,
   causing fixture cleanup errors, consumer failure and Redis-dependent skips.
   The test override uses `tmpfs: [/data]` without ownership/mode options.
   Inspect the inherited Redis user from the hardening baseline and fix the
   test tmpfs permissions, or explicitly disable persistence for this ephemeral
   test Redis. Production persistence must remain unchanged.

The latest harness teardown completed; `docker ps --filter name=paperless-ai-test`
returned no containers during the user-requested status check.

Loki was queried at `https://loki.docker.home.arpa/loki/api/v1/query_range`, using
past-30-minute windows and selectors:

```logql
{job="docker",host="docker.home.arpa",service_name="paperless-ai-test/webserver"}
{job="docker",host="docker.home.arpa",service_name="paperless-ai-test/broker"}
```

The relevant date is 2026-09-27, around **18:51–19:01 UTC**. Use absolute times
when resuming later, rather than a short relative window. For Phoenix startup
diagnosis, use the same labels with service name `paperless-ai-test/phoenix`,
around 18:40–18:46 UTC.

## Prepared pilot script — not yet executed

`tmp/language-pilot/pilot.py` is an ignored, one-off operational script in the
shared checkout. It has **not** been run or validated against production.
Review it before use. It imports the installed worker config/client/queues and
is intended to run inside the updated existing `paperless-ai-ai-1` container.

Modes:

- `snapshot`: fetches exactly the requested newest 10, asserts they are not
  already pending OCR/metadata work, records full API documents including OCR
  text and SHA-256 plus lookup resources, and writes
  `/tmp/language-pilot-before.json` inside the container.
- `enqueue`: reads that snapshot, verifies selected documents still exactly
  match it, then calls `TaskQueues.enqueue_metadata` for each fixed ID.
- `wait`: observes ready and delayed metadata queues for those IDs for up to
  45 minutes, records audit refresh/status/content hashes and changed field
  names, and writes `/tmp/language-pilot-after.json`. It does not execute a
  second worker or bypass retries.

Suggested sequence after tests and deployment succeed:

```bash
# On docker.home.arpa, in /opt/docker/paperless-ai; long operations use wake-run.
DOCKER_DOMAIN=docker.home.arpa docker compose up -d --build --no-deps --wait ai
docker cp tmp/language-pilot/pilot.py paperless-ai-ai-1:/tmp/language-pilot.py
docker exec paperless-ai-ai-1 python /tmp/language-pilot.py snapshot
docker cp paperless-ai-ai-1:/tmp/language-pilot-before.json tmp/language-pilot/before.json
# Do not enqueue until the before snapshot is safely copied outside the container.
docker exec paperless-ai-ai-1 python /tmp/language-pilot.py enqueue
docker exec paperless-ai-ai-1 python /tmp/language-pilot.py wait
docker cp paperless-ai-ai-1:/tmp/language-pilot-after.json tmp/language-pilot/after.json
```

Keep a copy of the after snapshot even if `wait` exits nonzero. Inspect failures
and unknown results, compare actual before/after metadata values, manually
review language agreement with stored text, and produce the per-document pilot
report. No before/after files exist yet. The operational files will contain
private document content; leave them out of Git. `.codex-wake-run/` is currently
untracked too; do not accidentally commit run logs.

## Work conventions to preserve

- Follow repository AGENTS.md and the user-provided defaults in the prior
  session: minimal changes, meaningful checks, preserve unrelated edits.
- Python work uses `uv`, with formatting from each affected Python project.
- Main-thread commands expected to exceed 10 seconds must use the installed
  **wake-run** skill. Read its instructions on the new host. Its normal contract
  is to end the turn after arming and resume on completion, without polling.
  The original user explicitly requested a status check after ten minutes;
  one check revealed the already-completed latest test run.
- Subagents may run their own bounded checks directly; do not delegate merely
  because a command is long. No agents from the old session are still working.
- Prefer centralized Loki for homelab logs. Never expose credentials.
- The user is frustrated by repeated long full-harness attempts. Diagnose the
  actual task response and Redis permissions first; avoid another sequence of
  five two-minute timeouts. Then run the required Docker harness and continue
  to deployment/pilot once verification is sound.

The original session stops after saving this document; resuming the task belongs
to the new session on docker.home.arpa.
