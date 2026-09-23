# Conversation history and lifecycle

## Purpose

Upgrade the existing durable-history UI into the prototype sidebar without redesigning completed #15. This file is the authoritative consumer description of the current conversation/message HTTP contract.

## In scope

New/list/select/rename/delete interactions, list grouping, initial restoration, draft protection, request races, loading/empty/error states, and the archive footer. History behavior lives in the existing inline JavaScript.

## Existing code to inspect

- `ai/src/paperless_ai/search/webhook.py`: `apiPath`, `api`, `renderConversationList`, `refreshConversations`, `createConversation`, `loadConversation`, startup promise chain and lifecycle button listeners.
- Same file: `_chat_title`, `_chat_owner`, `_require_chat_store`, all `/conversations` routes, `_restore_chat_sources`, `health`.
- `ai/src/paperless_ai/search/chat_store.py`: all CRUD methods, `_conversation`, `append_turn`.
- `ai/tests/test_chat_history.py`: `test_chat_store_persists_owned_conversations_and_sources`.
- [01](01-conversation-shell-and-responsive-layout.md): sidebar shell and drawer focus ownership.

## Implementation instructions

### Existing persistence API: preserve exactly

Routes below are FastAPI paths. Use the existing `apiPath()` to add `/ai` in the deployed page. Requests use same-origin authentication; never send an owner ID from browser state.

| Operation | Request | Success | Existing error behavior |
| --- | --- | --- | --- |
| Create | POST `/conversations`; empty object or optional `title` | 201, conversation summary | Invalid supplied title 422; uninitialized store 503 |
| List | GET `/conversations` | 200, object containing `items`, an array of summaries sorted by descending `updated_at` | Uninitialized store 503 |
| Load | GET `/conversations/{id}` | 200, summary plus ordered `messages` | Unknown/other-owner ID 404 |
| Rename | PATCH `/conversations/{id}` with `title` | 200, updated summary | Invalid non-null title 422; unknown/other-owner ID 404 |
| Delete | DELETE `/conversations/{id}` | 204, no body | Unknown/other-owner ID 404 |

A summary contains string `id` (UUID), string `title`, and ISO timestamps `created_at`, `updated_at`. It does not expose `owner_id`, message count, pagination cursor or preview text. The default title is “New conversation”; the first successfully appended turn replaces that exact default with the trimmed prompt's first 120 characters. Renaming and appending update ordering. Do not generate competing client titles.

A loaded message contains string `id`, `role` (`user` or `assistant`), string `content`, ordered `tool_activity` array, nullable string `model`, nullable `usage` object, ISO `created_at`, and `sources` array. Messages arrive in database position order; position is not exposed. Render in returned order, not timestamp order. The usage keys are `prompt_tokens`, `completion_tokens`, `total_tokens` when supplied. User messages also have these fields, normally empty/null. Tool records use the completed-tool payload fields described in [03](03-turn-rendering-tool-progress-and-composer.md). Source fields and refreshed availability are defined in [04](04-document-inspector-and-source-cards.md).

`append_turn` transactionally saves a user/assistant pair and sources. Failed/in-flight turns do not appear in restored history. REST message IDs differ from ephemeral WebSocket `turn_id`; do not merge them by ID or persist a browser-side transcript.

Known current boundary gaps: malformed UUIDs are passed to PostgreSQL, and PATCH with missing/null `title` passes `None` to a non-null database column. The bounded corrections and expected responses belong to [05](05-backend-contracts-and-migration-boundaries.md). Until that gate is delivered, these are not claimed as existing 422 behavior. `api()` must accept 204 and fall back to a generic error if a response is not JSON.

### Initial load and sidebar presentation

1. Start with history loading and send/lifecycle mutations unavailable. Distinguish history readiness from socket readiness. A connected socket alone must not enable sending into an unknown conversation.
2. List conversations; load the newest when nonempty. When empty, create one durable empty conversation and select it, preserving current startup behavior. A failed list must never be interpreted as an empty list or trigger creation. Show Retry.
3. New chat creates a durable record immediately. On success, clear transcript, turn map and selected source/iframe, update sidebar, and focus the prompt. If already in an empty selected conversation with an empty draft, reuse it rather than creating another blank record. Prevent double-click creates while the request runs.
4. Group the returned list by browser-local `updated_at`: Today, Previous 7 days (preceding seven local calendar dates), and Older. Hide empty groups. Keep server ordering inside each group. Display local time for Today and a concise date otherwise, with a full localized timestamp accessible. Do not add a Yesterday group absent from this chosen grouping.
5. An active entry has the soft-green fill and accessible current state. Long titles truncate visually but retain full accessible text. Keep rename/delete controls for the active conversation only; do not nest buttons inside the selection button. Empty conversations show their server title and an empty-center invitation.

### Selection, drafts and race rules

Use one selected conversation and one active turn at a time. [03](03-turn-rendering-tool-progress-and-composer.md) owns the shared busy/connection rules. While a turn is pending/running, disable New, selecting a different conversation, rename and delete; keep read-only source inspection available. Explain the temporary disabled state in text.

On selection, preserve the current transcript until the requested history succeeds; show a loading indicator and disable send. Only the latest selection response may replace the selected ID/transcript. Ignore superseded loads and sidebar refreshes; loading A then B must not allow a late A response to overwrite B. Changing conversations clears the preview through the inspector reset hook.

An unsent draft belongs only to the active conversation in memory. Before New, switching or deleting the active conversation, confirm discarding a nonempty draft; cancel leaves everything untouched. Do not persist drafts or conversations to browser storage. Rename does not discard drafts. Successful submission clears only the submitted text; a draft typed while a turn runs remains intact on completion.

For restoration, reuse the same answer/tool/usage/source render functions as live turns. Replayed turns are complete, not “Waiting…” or running. Empty `sources`, null usage and missing optional metadata are valid. After loading, bring the latest message into view once, without repeatedly overriding manual scrolling.

### Rename, delete and failure states

- Rename opens an accessible, prefilled single-field dialog. Trim input; require 1–120 characters, matching server validation. Save only on confirmation; Escape cancels; keep dialog/input on failure. Successful PATCH updates the sidebar from the response, with a subsequent refresh if needed. Prevent duplicate submissions.
- Delete confirms the conversation title and states that only chat history is deleted, never Paperless documents. On 204, clear selected preview/transcript, refresh the list, load its newest entry, or create a fresh empty conversation when none remain. If replacement loading/creation fails, show no selected conversation and disable send with Retry; never retain the deleted ID.
- On load/rename/delete 404, explain that the conversation is unavailable, refresh the list and let the user choose a remaining entry. Do not silently redirect a failed mutation to another conversation. Treat a raced delete 404 as already unavailable and clear the stale selection.
- List/create/load failures retain usable existing content and show a scoped retry/error message; no indefinite spinner, fabricated success or automatic mutation retry. Retrying a create after an uncertain network result first refreshes the list so the user can see whether creation succeeded.
- After a successful completed turn, refresh titles/order without replacing the live transcript. Other-device changes become visible on reload or explicit history Refresh; no polling/synchronization engine is required.

### Archive status

Keep a footer below history with “Archive status”. Consume existing prefix-aware GET `/health` on initial page load and explicit retry/Refresh; this is separate from socket connectivity. Its body contains `status`, stage `pending` counts and `worker`, and may use HTTP 503 with a valid degraded body. Do not route it through a helper that discards that body on non-2xx.

Show “Processing available” for `status: ok`, “Processing degraded” for a degraded response, and “Status unavailable” on network/invalid response. An optional secondary line may sum the actual pending stage counts, explicitly labelled “pending tasks”, not documents. Omit indexed count and last-index timestamp per [00](00-overview-and-dependencies.md); worker heartbeats and pending tasks do not establish either. This footer must not prevent chat use when history and socket are ready.

## Acceptance criteria

- Reload and another authenticated device restore server conversations, messages and sources; no local-history fallback exists.
- Empty startup/New creates a durable record only after a successful request; failures leave a recoverable UI.
- Rename validates whitespace/length, respects cancellation and updates ordering; deletion never deletes a Paperless document.
- Rapid A/B selections, double New clicks, active-turn navigation and a dirty draft cannot mix transcripts or misassign a prompt.
- Restored turns use the shared live renderers, with null/empty fields handled and selected preview reset.
- Archive footer never claims an index count or timestamp unsupported by the existing API.

## Verification

From `ai/`, run `uv run ruff format .` after inline page changes. Use `./run_tests.sh` from the repository root for PostgreSQL/API integration, as specified in [06](06-test-and-verification-plan.md); a local skipped `test_chat_history.py` is not a persistence pass. In the browser exercise empty/new/rename/delete, slow out-of-order loads, two owners, 404/422/503, offline history, drafts and reload. Use faithful contract responses for isolated UI checks, never a new storage layer.

## Non-goals

No database/schema changes, browser history storage, pagination/search/archive features, cross-device real-time synchronization, server-generated summaries, transcript editing or replaying durable history into model context.

## Dependencies

- Implemented external prerequisite: #15, documented in [00](00-overview-and-dependencies.md).
- Start after [05](05-backend-contracts-and-migration-boundaries.md) and [01](01-conversation-shell-and-responsive-layout.md).
- Expose selected-ID/history-ready/lifecycle-loading hooks for [03](03-turn-rendering-tool-progress-and-composer.md); consume its busy state when integrated.
- Source reset/render contract: [04](04-document-inspector-and-source-cards.md). Verification: [06](06-test-and-verification-plan.md).
