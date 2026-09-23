# Turn rendering, tool progress and composer

## Purpose

Present the existing turn-based WebSocket interaction in the prototype's center column while keeping message routing, completed-turn persistence and rendering safe during loading, failure and reconnect.

## In scope

Live/restored turn rendering, status/tool timeline, sanitized assistant Markdown, composer shortcuts, one-turn-at-a-time controls, connection state, explicit reconnect, usage and source-render handoff. This file owns the WebSocket consumer contract.

## Existing code to inspect

- `ai/src/paperless_ai/search/webhook.py`: `chat_ws`, socket listeners, submit/keydown listeners, `createTurn`, `getTurn`, `addUserBubble`, `addTimelineItem`, `updateTool`, `getOrCreateToolCard`, `renderMarkdown`, `setAssistantMessage`, `setTurnError`, `updateUsage`, `scrollConversation`, `autosizePrompt`.
- `ai/src/paperless_ai/search/chat_agent.py`: `ChatCopilot.run_turn`, `_emit`, `_merge_source_flags`, `ChatTurnResult`.
- `ai/src/paperless_ai/search/tools.py`: `ToolExecutionResult`, `execute_tool_call_detailed` and summaries from metadata/search/read tools.
- `ai/tests/test_chat_tools.py`: `test_chat_copilot_run_turn_emits_events_and_aggregates_usage` and tool/source tests.
- [02](02-conversation-history-and-lifecycle.md): selected conversation, readiness and restored-message contracts.

## Implementation instructions

### Existing WebSocket contract

Connect to prefix-aware `/ws/chat`, using `wss` under HTTPS. A submitted message has `content` and `conversation_id`; the redesign always supplies the selected durable UUID. Keep the server's legacy plain-text/omitted-conversation support intact, but do not rely on it to bypass failed history initialization.

| Incoming event | Fields beyond `type` | Rendering/meaning |
| --- | --- | --- |
| `conversation_created` | `conversation` summary, no `turn_id` | Existing legacy fallback: adopt the returned ID only for the pending creation flow; refresh history |
| `turn_started` | `turn_id`, `conversation_id` | Associate the pending submitted prompt with a server turn and its selected conversation |
| `status` | `turn_id`, `phase` (`model` or `tool`), `content` | Show concise current activity; do not duplicate completed-tool text as another timeline row |
| `tool_call_started` | `turn_id`, `tool_call_id`, `name`, `arguments` | Create/update running activity for this turn/tool |
| `tool_call_completed` | Prior tool fields plus `summary`, `preview`, `duration_ms` | Finish activity, show actual summary, retain detail disclosure |
| `usage` | `turn_id`, `scope` (`step` or `total`), `model`, `available`, optional token counts | Display final `total`; do not add step counts to a total already aggregated by the server |
| `assistant_message` | `turn_id`, `content` | Render final sanitized answer; this is not a token delta |
| `sources` | `turn_id`, `items` | Delegate to [04](04-document-inspector-and-source-cards.md) |
| `error` | `turn_id`, `content` | Known-turn failure or connection/request-level error; see sentinel IDs below |
| `turn_completed` | `turn_id`, `success` | Mark terminal saved success or failure, release busy controls, refresh history on success |

The successful server sequence is: start → agent status/tool/step usage → source metadata build → `append_turn` transaction → answer → total usage → sources → completed true. An agent/save failure emits error then completed false. Sources and final answer are therefore only published after a successful save. Do not label a turn saved merely because a tool finished.

`turn_id: invalid` denotes invalid input or inaccessible conversation and can arrive without start/completion. `turn_id: unavailable` denotes setup failure followed by close code 1011. Handle these as request/service errors, not ordinary persisted turns. Other events normally have only `turn_id`, so retain the mapping established at `turn_started`. Unknown event types can be ignored; invalid JSON must show a recoverable connection error without crashing all listeners.

There are no answer-token deltas, cancel commands, event replay, client idempotency keys, heartbeat messages or server message IDs in this protocol. Preserve the existing protocol; any additions require [05](05-backend-contracts-and-migration-boundaries.md) to be revised first.

### State and message ownership

Use the existing page-level state and `turns` map; do not introduce a state-management library. Keep socket readiness, selected-history readiness and active-turn state separate.

| State | Required UI behavior |
| --- | --- |
| Connecting / history loading | Composer text remains editable; send disabled; status describes the missing readiness |
| Ready | Send enabled only for nonblank input, open socket, selected durable conversation, completed history load and no active mutation/turn |
| Submitted, awaiting start | Immediately reserve the active turn and show the optimistic user prompt/pending activity; block duplicate submission and conversation mutations |
| Running | Tool progress and current status visible; prompt can hold the next draft, but send stays disabled |
| Completed successfully | Final answer, sources and usage visible; refresh sidebar; release busy state without changing the next draft |
| Failed with a terminal server error | Preserve failed prompt/activity in the current view as unsaved; show actionable failure and permit explicit correction/resubmission |
| Disconnected during a turn | Mark outcome unknown/interrupted, not cancelled or definitively unsaved; retain visible work and draft until reconciliation |

Only accept a start for the selected conversation's pending request, and only route subsequent events to its known turn. Do not let `getTurn()` silently create a new turn for late/unrelated events. Socket listeners from replaced connections must be retired or ignored. History selection resets turn and source state through the shared hooks from [02](02-conversation-history-and-lifecycle.md) and [04](04-document-inspector-and-source-cards.md).

### Tool activity and answers

Place the visible activity list above the answer, as in the prototype. Show compact running indicator/completed marker, readable tool label and returned summary. Map the three existing names to “Check archive metadata”, “Search documents” and “Read document”; unknown names remain readable text. Use actual completion summaries for counts/documents, not the prototype's fixed “18” or “3”. A completion event means the tool returned, not that a factual claim was verified; retain warnings/failure text in its summary.

Keep arguments, output preview and duration under an accessible per-tool disclosure. Tool arguments/previews remain plain text. Identify calls by turn and `tool_call_id`; for restored records with absent IDs, use their stable per-message array position so they do not collapse into one card. Preserve the tool array order. If a turn fails while a tool is running, remove the spinner and mark that activity interrupted, without manufacturing a completion result.

User bubbles align right with soft-green fill. Assistant turns align left with a small assistant marker and bordered white answer. Use `marked.parse` with GFM and `DOMPurify.sanitize` for both live and restored answers. Never insert user content, source metadata or tool output as raw HTML. If Markdown dependencies fail to load, display the answer as plain text with a nonblocking notice; sending must not leave an invisible answer. Preserve safe links and accessible overflow for tables/code. Do not invent link citations by guessing numbers in prose; source interaction belongs to [04](04-document-inspector-and-source-cards.md).

Show sources expanded under the answer when present, with the count of actual source items, not an asserted count of documents supporting every claim. Retain model and final token usage as subdued secondary details after sources. Display the model even when usage is unavailable, with “Tokens unavailable”; do not imply unknown counts are zero. Restored messages use the same renderers with stored fields and no running animation.

Show stored `created_at` for restored user/assistant messages. Live receipt times may be shown as local display times, without claiming a server timestamp; tool events have duration but no stored wall-clock timestamp. Omit exact tool times rather than synthesizing historical ones.

### Composer and scrolling

Use a labelled multiline textarea inside the white composer panel, a circular green Send button, and keyboard help. Keep the existing Enter-to-send behavior; Shift+Enter inserts a newline, and Ctrl+Enter/Command+Enter also sends. Do not send during IME composition. Show help matching these rules instead of copying only the screenshot's Command+Enter hint.

Autosize from roughly 44px to a maximum of 160px (or less when needed to keep a short viewport usable); then scroll inside the textarea. Reset its height after successful submission. Disable send for whitespace, disconnected state, loading and active turn; check these conditions in the submit handler as well as the button. If `socket.send` throws, retain the text and avoid leaving a duplicate optimistic user bubble. Do not clear a newly typed draft when an older turn completes.

Auto-scroll for the user's own submit and when the reader was already within 64px of the transcript bottom. When the reader scrolls up, preserve position as tool/answer/source updates arrive; offer “Jump to latest”. Completion and source selection must not steal keyboard focus. Empty conversations show a brief invitation under the heading, not fictional answers or prompts submitted automatically.

### Cancel and reconnect policy

The current server awaits an entire turn before reading the next message. Closing a socket is not a reliable cancellation operation and may race with persistence. First delivery therefore has no Stop/Cancel generation control. Escape closes dialogs/drawers; it never claims to cancel inference. True cancellation is the unresolved extension listed in [00](00-overview-and-dependencies.md).

Replace the current reload-only error instruction with an explicit Reconnect action that creates a fresh socket, retaining the in-memory draft and selected ID. Do not reconnect in a tight automatic loop and never resend a prompt automatically. After connection recovery, reload the selected durable conversation before enabling send; replace rather than merge restored messages by text or timestamps. Show that socket recovery resets model context, even though saved messages remain visible.

If disconnection happened during a turn, keep its submitted text in a separate visible recovery area until reconciliation; do not overwrite a next draft. Reload may show a committed pair, or may occur before a still-running server task saves. State “The previous response may still be saving. Refresh history before resending.” Provide explicit history refresh and an explicit user choice to resend, warning that it may duplicate a completed request. No automatic retry or claim of exactly-once delivery is possible with the current protocol. A normal failed terminal turn can offer “Use this prompt again”, restoring its text only after confirming replacement of a nonempty draft.

Place a concise notice on restored/switched conversations and after reconnect: “Saved messages are shown here. Earlier messages are not included in a new session's AI context.” This reflects the existing server reset; do not add stored-message replay to hide it.

## Acceptance criteria

- Actual start/status/tool/answer/usage/source/completion events produce one correctly routed turn; no token-stream assumption exists.
- Rapid Enter/clicks produce one send, active-turn navigation is blocked, and next-draft text survives completion.
- Completed-tool summaries are visible without opening developer details; restored calls with missing IDs remain distinct.
- Malicious Markdown/metadata cannot execute script; dependency failure still yields readable plain text.
- Save failures never show saved success; sentinel errors do not create bogus turns or lock controls indefinitely.
- Disconnection preserves recovery text, reconnect reloads history and never automatically resends or claims cancellation.
- Usage is not double-counted and model remains visible when token counts are unknown.

## Verification

From `ai/`, run `uv run ruff format .` and `uv run pytest tests/test_chat_tools.py -q`. Add the bounded socket/renderer-state checks described in [06](06-test-and-verification-plan.md) during implementation; they are absent from the current suite. Browser checks must exercise all event/error states, IME/keyboard sending, scrolling, disconnect before/after persistence, and restored Markdown. Use deterministic model mocks; existing test fixtures do not automatically patch the separate running copilot container.

## Non-goals

No protocol rewrite, token streaming, generation cancellation, automated retries, persisted drafts, optimistic database writes, background turns across selected chats, agent/tool changes or replay of durable history into the model.

## Dependencies

- Start after [01](01-conversation-shell-and-responsive-layout.md), [02](02-conversation-history-and-lifecycle.md) and [05](05-backend-contracts-and-migration-boundaries.md).
- Delegate source markup/preview and reset to [04](04-document-inspector-and-source-cards.md).
- Capability decisions: [00](00-overview-and-dependencies.md). Verification: [06](06-test-and-verification-plan.md).
