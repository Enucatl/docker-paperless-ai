# Chat Copilot redesign: overview and dependencies

## Purpose

Specify the production `/ai/chat` redesign requested by [issue #16](https://github.com/Enucatl/docker-paperless-ai/issues/16). These eight files are implementation instructions for subsequent Luna agents; this delivery contains no application implementation.

Evidence baseline: checkout `d6d1e75f489640cf48fe4951515fd0244259be5c`, inspected on 2026-09-23. [Issue #15](https://github.com/Enucatl/docker-paperless-ai/issues/15) closed on 2026-09-21 at 20:56:34 UTC; commit `304aa91` already implements persistence. Treat it as an implemented external contract to preserve, not future infrastructure to invent.

The [reference image](https://raw.githubusercontent.com/Enucatl/docker-paperless-ai/design/chat-copilot-prototype/docs/assets/chat-copilot-prototype.png) was inspected directly. It is versioned on `design/chat-copilot-prototype`, at `docs/assets/chat-copilot-prototype.png`. Keep that reference external to this package; do not copy the prototype or its fictional documents into product assets. GPT-6 Astra produced this package after inspecting the repository and reference image.

## In scope

An ordinary HTML/CSS/JavaScript redesign of the existing inline page: green brand bar, durable conversation sidebar, visible tool activity, readable answers and sources, bottom composer, and selected-document inspector. Preserve working backend and Paperless flows. Use the reference for visual hierarchy and these specifications for behavior absent from the image.

## Existing code to inspect

Paths below are relative to the repository root.

| Evidence | Relevant behavior |
| --- | --- |
| `ai/src/paperless_ai/search/webhook.py`: `chat_ui` | Inline HTML, styles and JavaScript; Paperless static CSS, marked and DOMPurify; no separate frontend build |
| Same file: `apiPath`, `api`, history functions and listeners | Prefix-aware CRUD and basic list/select/new/rename/delete already work |
| Same file: `chat_ws` | Sequential turns, `conversation_id`, persistence before final answer, socket-local model context |
| Same file: `_build_chat_sources`, `_restore_chat_sources`, `_document_*_url` | Live metadata, historical snapshots and same-origin Paperless URLs |
| `ai/src/paperless_ai/search/chat_agent.py`: `ChatCopilot.run_turn`, `ChatTurnResult` | Status/tool events, aggregated usage, matched/read references |
| `ai/src/paperless_ai/search/chat_store.py`: `ChatStore` | Owner-scoped PostgreSQL CRUD and atomic completed-turn persistence |
| `ai/src/paperless_ai/search/migrations/0001_chat_history.sql` | Existing schema and cascading deletion; read only for this redesign |
| `common/src/paperless_common/paperless.py`: chat metadata helpers | Metadata fields and name resolution |
| `docker-compose.yml`: `ai` routing; `README.md`: Copilot conversation database | `/ai` prefix stripping, authentication middleware, database and owner-header configuration |
| `ai/tests/test_chat_history.py`, `test_chat_tools.py`, `test_paperless_client.py`, `test_webhook_startup.py` | Existing verification evidence; see [06](06-test-and-verification-plan.md) for coverage limits |

## Implementation instructions

### Target architecture and preserved behavior

Keep `chat_ui()` as the serving boundary and use semantic elements and named JavaScript functions within its existing inline document. No frontend framework, package manager or bundler is warranted. Preserve existing IDs where practical so serial agents can improve different behaviors without breaking selectors.

The browser uses authenticated, same-origin `/ai/conversations` and `/ai/ws/chat`; FastAPI receives unprefixed paths. Paperless links remain root-relative `/documents/...` and `/api/documents/...`, never `/ai/api/documents/...`. Preserve unprefixed `/chat` API path derivation for direct service use, while recognizing that Paperless assets/previews need its origin or the production proxy.

| Preserve | Required outcome |
| --- | --- |
| Completed conversation history | Lists and restores durable user/assistant messages, tool activity, model/usage and source references across reloads/devices |
| Turn execution | One active turn in the redesigned UI; status and tool activity remain visible while awaiting the final answer |
| Markdown | GFM tables, lists, links and code render through existing marked plus DOMPurify; user text remains plain text |
| Source provenance | Distinguish matched and inspected documents; retain document IDs, metadata, thumbnails and full Paperless links |
| Preview | Keep the Paperless preview endpoint in an iframe; browser/Paperless supplies PDF controls |
| Source refresh | Reopening history refreshes current metadata; missing documents retain an unavailable historical reference |
| Runtime | Preserve lazy search warmup, agent/tool/search behavior, telemetry and existing worker health behavior |

Do not confuse restored display history with model memory. `chat_ws` resets its in-memory `history` on conversation changes and a new socket starts empty. #15 deliberately does not replay stored history into the model. [03](03-turn-rendering-tool-progress-and-composer.md) owns communicating this limitation.

### Delivery sequence and dependency map

Read [07](07-luna-agent-handoff.md) before assigning work. Execute serially because several agents edit `webhook.py`:

1. **00**: establish baseline and scope.
2. **05**: verify existing contracts and deliver only the narrowly identified backend adjustments.
3. **01**: establish shell, tokens, responsive containers and accessibility structure.
4. **02**: connect history/lifecycle controls to the preserved CRUD contract.
5. **03**: connect turn rendering, socket state and composer to the shell/lifecycle state.
6. **04**: complete source cards and inspector using the preceding render hooks.
7. **06**: close verification gaps and perform integrated visual/interaction checks.
8. **07**: apply the final handoff/review checklist.

Contract dependencies are explicit: [02](02-conversation-history-and-lifecycle.md) owns the CRUD/message contract; [03](03-turn-rendering-tool-progress-and-composer.md) owns the WebSocket event contract; [04](04-document-inspector-and-source-cards.md) owns source presentation; [05](05-backend-contracts-and-migration-boundaries.md) owns permitted backend changes. Each agent consumes the prerequisite spec instead of redefining it.

### Boundary with completed #15

Consume `ChatStore` and the existing endpoints now. No waiting for a new datastore or API design is required. A checkout missing those interfaces must first integrate #15; UI-only development may mock its exact responses, but must not ship the mock or a browser-history substitute. A database/proxy configuration failure blocks integration verification, not visual work against faithful mocks.

Do not alter schema, migrations, database role/secrets, transaction boundaries, ownership semantics or prompt-context persistence. The narrow request validation and source-refresh adjustments in [05](05-backend-contracts-and-migration-boundaries.md) are HTTP/presentation-boundary work, not a persistence redesign.

### Unresolved details and binding first-delivery decisions

| Prototype detail / evidence gap | First delivery | Decision needed only to expand scope |
| --- | --- | --- |
| Indexed count and last-index time | Archive footer reports existing worker health; omit unsupported count/time | Authoritative indexing-statistics contract and meaning of “indexed” |
| User name, initials and account dropdown | Omit personal/account controls; keep connection status and accessible help | Trusted display-profile source and supported account actions |
| Attachment paperclip | Omit upload control; rich composer means multiline input, send and keyboard help | Upload/attachment intent and endpoint |
| “High relevance” and “Relevant” | Show actual Matched / Read in full flags; no invented scores | Calibrated relevance field and label thresholds |
| Cancel generation | No Stop/Cancel control; reconnect never claims cancellation | Server cancellation semantics, including save races |
| Historical follow-up memory | Explicit notice when context is reset | Separate product request to replay history; excluded by #15 |
| “Added” date | Add a nullable live metadata field per [05](05-backend-contracts-and-migration-boundaries.md); show “Not available” if absent | Verify supported Paperless response supplies `added` |
| Document access across different users | Preserve current service-token backend and Paperless browser authentication; verify proxy ownership behavior | Deployment must confirm archive sharing assumptions; per-user document retrieval is a separate authorization design |

These decisions allow implementation without pretending the screenshot supplies backend capabilities. Record any later product resolution in the relevant spec/handoff before expanding scope.

## Acceptance criteria

- All eight requested files exist with the required sections and working relative dependencies.
- Every current contract is distinguished from proposed changes; #15 is identified as completed.
- Prototype structure is traceable to [01](01-conversation-shell-and-responsive-layout.md); unsupported controls/data have explicit behavior above.
- Each implementation agent has bounded ownership, acceptance criteria and a verification path.
- This specification delivery changes only the eight Markdown files in this directory.

## Verification

For this package, run `git diff --check`, list the directory with `find docs/implementation/chat-copilot-redesign -maxdepth 1 -type f -name '*.md' | sort`, and inspect `git status --short --untracked-files=all` from the repository root. Verify eight files, required headings, referenced symbols and relative links. Application tests are not evidence of a documentation-only change and need not run now.

For subsequent implementation, use [06](06-test-and-verification-plan.md); do not claim browser or integration acceptance merely because the package was reviewed.

## Non-goals

No application code in this delivery; no commits. No replacement frontend, persistence, database migrations, uploads, chat-history search, conversation archiving, collaboration, advanced accounts, model-memory replay, custom PDF viewer, retrieval changes or fabricated demo data in production.

## Dependencies

- External implemented prerequisite: [#15](https://github.com/Enucatl/docker-paperless-ai/issues/15).
- Visual/task authority: [#16](https://github.com/Enucatl/docker-paperless-ai/issues/16) and its linked prototype.
- Package execution: [07](07-luna-agent-handoff.md); verification: [06](06-test-and-verification-plan.md).
