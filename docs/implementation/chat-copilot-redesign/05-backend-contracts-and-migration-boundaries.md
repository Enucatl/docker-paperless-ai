# Backend contracts and migration boundaries

## Purpose

Freeze the implemented #15 interfaces and identify the few backend boundary adjustments needed for the redesigned UI. This is the first implementation work item after the overview, despite its filename number.

## In scope

Verify compatibility of existing routes/events, validate malformed conversation inputs at the transport boundary, add the inspector's optional live Added field, and isolate expected per-document metadata failures. Specify authentication/authorization limits without introducing new storage or a competing API.

## Existing code to inspect

- `ai/src/paperless_ai/search/webhook.py`: `lifespan`, `_chat_owner`, `_chat_title`, `_require_chat_store`, conversation endpoints, `chat_ws`, `_build_chat_sources`, `_restore_chat_sources`, `_document_*_url`, `health`.
- `ai/src/paperless_ai/search/chat_store.py`: owner predicates, transaction and append semantics; do not edit for the redesign.
- `ai/src/paperless_ai/search/migrations/0001_chat_history.sql`: read only to verify compatibility.
- `common/src/paperless_common/paperless.py`: `get_document_for_chat`, `get_document_chat_metadata`, `_raise_for_status` and existing niquests request behavior.
- `docker-compose.yml`: database wiring and `ai` Traefik labels; `db/init/ai-chat-user-db.sh`; `README.md`: Copilot conversation database. Read only.
- `ai/tests/test_chat_history.py`, `test_paperless_client.py`, `test_chat_tools.py`, `test_webhook_startup.py`, `conftest.py`; `docker-compose.test.yml` and `run_tests.sh`.

## Implementation instructions

### Preserve contracts; do not add endpoints

The exact CRUD/message shapes are in [02](02-conversation-history-and-lifecycle.md), and event shapes/order in [03](03-turn-rendering-tool-progress-and-composer.md). Implement those as consumers of the current server. Do not duplicate their contract tables here or rename fields for UI convenience.

Preserve route methods/statuses, optional create title, default title behavior, owner-filtered 404 responses, message ordering, 204 delete, and completed-turn save-before-answer semantics. Preserve legacy plain-text/omitted-conversation WebSocket requests for compatibility while the new UI always supplies the selected ID. Preserve nullable usage and the distinction between stored message IDs and transient turn IDs.

No new history/profile/archive/source endpoint or WebSocket event is needed for this delivery. `/metadata/available` is a tool/filter vocabulary endpoint, not profile or archive-statistics data. `/health` already reports worker/pending-task status; consume it according to [02](02-conversation-history-and-lifecycle.md), without extending its response to inferred index metrics.

### Narrow changes authorized for later implementation

| Change | Current evidence | Required contract after implementation |
| --- | --- | --- |
| Validate conversation IDs | REST path IDs and WebSocket `conversation_id` reach PostgreSQL as unchecked strings | Malformed UUID in GET/PATCH/DELETE returns 422, without database work. Malformed WebSocket ID emits existing `error` with `turn_id: invalid`, no turn start/save, socket stays usable. Well-formed unknown/foreign IDs retain 404 or existing “Conversation not found” socket behavior. |
| Require rename title | `_chat_title` allows missing/null title for POST, but PATCH forwards it to a NOT NULL column | PATCH missing/null/blank/non-string/>120-character title returns 422. POST missing/null optional title continues creating the default. Preserve trimming and the existing valid-title length limit. |
| Expose Added metadata | `get_document_for_chat` fields omit `added`; metadata helper omits it too | Request Paperless `added` and return it as nullable `added` from `get_document_chat_metadata`. Carry it through existing source assembly/restoration. No schema migration or rewrite of old source rows. |
| Isolate unavailable sources | `_restore_chat_sources` handles `None`/404 snapshots, but expected HTTP/transport exceptions can fail the whole load; `_build_chat_sources` currently drops missing documents | On expected per-document access/not-found/temporary metadata failures, keep the source ID and flags with `available: false`; retain the owned historical snapshot on restoration, or use only ID/fallback title for a fresh source. Other sources/messages still render. Successful sources preserve existing shape; setting `available: true` is an additive optional normalization. |

Use existing validation facilities/stdlib and helper boundaries; do not introduce a new model/service layer merely for these checks. Inspect all callers before changing `_chat_title` so POST retains its optional-title contract. Validate transport input before invoking the store; database ownership still remains authoritative. Do not broaden this work into rewriting every WebSocket payload coercion or unrelated error handling.

The unavailable-source adjustment concerns expected niquests transport failures and HTTP access/not-found/server failures while obtaining that source's metadata. Do not catch cancellation or programming errors under a blanket success fallback. Use existing status/error inspection patterns such as `_is_retryable_paperless_error` where appropriate, after confirming its exact semantics; do not turn authentication failures into retries. Unavailable means “could not load”, not “definitely deleted”. The UI renders that distinction according to [04](04-document-inspector-and-source-cards.md).

For a missing fresh source, preserve the known ID and matched/inspected flags with canonical document URLs and a minimal unavailable label. This avoids silently losing a tool reference when a document disappears before final rendering. Persist only that display/reference data through the existing `append_turn`; no source-PDF copying or transaction change. A save failure must still fail the turn, even when source metadata failures are now represented gracefully.

Verify `added` against the Paperless version in the Docker harness. The baseline repository does not establish that field's deployed response. If unsupported, retain the nullable field/fallback and record the compatibility finding; never derive Added from `created`, conversation timestamps or a guessed date. Old stored snapshots remain valid because live restoration can supply the new field and absent data is allowed.

### Authorization and routing boundary

`_chat_owner` reads `CHAT_OWNER_HEADER`, default `Remote-User`, trims it and uses `None` for absent/blank. `ChatStore` filters with null-safe owner equality. Nullable ownership is the documented shared single-user mode, not a separate account per anonymous browser. Do not expose another owner's rows, accept a browser-supplied owner field, or cache one user's transcript for another login.

The reverse proxy must authenticate both HTTP and WebSocket upgrade requests and strip/replace spoofed identity headers. Current Compose routes `/ai` through prefix stripping, `authelia@docker` and `secured@file`; this checkout does not include the full external middleware configuration, so header sanitization is a deployment verification item, not proven by the labels. Leave this routing in place. No JavaScript-readable service tokens or credentials may be added to the page.

Ownership and document permissions are distinct. The AI uses its configured Paperless service token for metadata/search; the browser uses Paperless authentication for direct thumbnails/previews/details. The current implementation does not establish per-user Paperless ACL filtering of AI search/source metadata. Do not claim that conversation isolation solves that. The deployment owner must confirm the shared-archive assumption before asserting multi-user document isolation; any user-scoped Paperless credential design is outside this redesign.

Preserve root-relative Paperless URLs and prefix-aware AI paths; use canonical positive document IDs for browser preview destinations. Source refresh can show a saved snapshot owned by that conversation when a document is unavailable, labelled historical. Never bypass an unavailable browser preview by proxying the file using the more privileged AI token.

### Migration and persistence boundaries

#15 already owns database/role creation, `CHAT_DATABASE_URL`, password-secret loading, startup migrations, conversations/messages/reference tables, owner predicates, cascading deletion, transactional completed-turn writes and source snapshots. Its configuration is documented and deployed separately from this visual work.

No migration, table, index, alternate datastore, retention job, persisted UI selection or draft storage is required. Do not modify `chat_store.py`, SQL migrations, provisioning or Compose for this redesign. The Added field travels through existing JSON display metadata and live refresh; old rows require no backfill. Do not store raw PDFs/OCR/thumbnail bytes or replay stored messages into model context.

If the implementation checkout lacks #15, integrate it before this work. If the store fails configuration/startup, use exact contract mocks for UI development and report the integration blocker; do not silently degrade to localStorage. #15 being closed is evidence of delivery, not a substitute for checking its interfaces on the implementation revision.

### Compatibility and error expectations

Old clients still recognize all event types and existing fields. Additive nullable `added` and availability are safe for old snapshots/renderers. No new success can be emitted before `append_turn` completes. Agent/save failure retains error then unsuccessful completion; a deleted conversation during append remains unsuccessful.

Distinguish anticipated HTTP 422/404/503 from transport errors. `_require_chat_store` explicitly produces 503 only when uninitialized; runtime database exceptions are not currently normalized into that response, so the frontend must handle generic 5xx/non-JSON failures. Do not document every database outage as guaranteed JSON 503. Application/service errors remain visible without revealing new credentials or adding debug internals to product surfaces.

## Acceptance criteria

- Existing well-formed HTTP and WebSocket contracts remain compatible with #15; no schema/configuration changes are needed.
- Malformed UUIDs and invalid PATCH titles fail predictably before datastore operations; optional POST title behavior is retained.
- Foreign-owner well-formed IDs disclose no history through CRUD or socket selection.
- Added is nullable live metadata; unsupported/missing values do not break old conversations.
- Missing/inaccessible/temporarily failing source metadata does not hide unrelated restored messages or sources; source references remain intact.
- Tool/model failures and database save failures still report unsuccessful turns; unavailable source handling never swallows persistence failure.
- Deployment verification distinguishes conversation ownership, trusted proxy headers and Paperless document permissions.

## Verification

From `ai/`, run `uv run ruff format .` and `uv run pytest tests/test_chat_tools.py tests/test_paperless_client.py tests/test_webhook_startup.py -q`. After editing the shared helper, run `uv run ruff format .` from `common/` as well; run its consumer tests from `ai/`, where the test dependencies/lockfile live.

Add focused boundary tests during implementation as assigned by [06](06-test-and-verification-plan.md), then run them from `ai/`. Use `./run_tests.sh` from the repository root for PostgreSQL, real Paperless metadata and authenticated HTTP/WebSocket integration; preserve its isolated project/cleanup and report skips. Proxy identity checks require the deployed middleware or an equivalent isolated proxy configuration, not just unit mocks.

## Non-goals

No #15 reimplementation, migration/backfill, alternate database, authentication platform, new account/profile/index-statistics endpoint, custom document proxy, server cancellation, event replay, idempotency protocol or model-memory changes. Existing unrelated persistence/runtime defects are reported separately unless they prevent the bounded contract above.

## Dependencies

- [00](00-overview-and-dependencies.md) establishes scope and completed #15 baseline.
- Consume the API description in [02](02-conversation-history-and-lifecycle.md), WebSocket description in [03](03-turn-rendering-tool-progress-and-composer.md) and source requirements in [04](04-document-inspector-and-source-cards.md) as design references; those UI implementations need not exist before this work.
- Deliver this implementation before [01](01-conversation-shell-and-responsive-layout.md)–[04](04-document-inspector-and-source-cards.md).
- Verification ownership and handoff: [06](06-test-and-verification-plan.md), [07](07-luna-agent-handoff.md).
