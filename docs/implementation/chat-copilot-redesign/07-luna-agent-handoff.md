# Luna agent handoff

## Purpose

Provide a strict sequence and ownership checklist for independent Luna agents implementing this package one file at a time. Read this file before starting; use it again as the final acceptance checklist.

## In scope

Preflight, ordered assignments, edit boundaries, evidence handoff and handling of missing dependencies/unresolved details. This specification does not authorize application implementation during the documentation-only delivery that created it.

## Existing code to inspect

- Repository `AGENTS.md` and any future nested instructions, plus `git status` and current revision before each assignment.
- `ai/src/paperless_ai/search/webhook.py`: shared edit surface for backend handlers and inline UI; inspect prerequisite agents' actual changes before editing.
- `ai/src/paperless_ai/search/chat_store.py`, `migrations/0001_chat_history.sql`: existing #15 contract, read only.
- `common/src/paperless_common/paperless.py`: only backend agent owns any bounded metadata change.
- `ai/tests/`, `run_tests.sh`, `docker-compose.test.yml`: evidence and environmental constraints per [06](06-test-and-verification-plan.md).

## Implementation instructions

### Preflight for every agent

1. Read your assigned spec and linked prerequisites. Verify the implementation revision still has the recorded interfaces; the package baseline is identified in [00](00-overview-and-dependencies.md), not assumed forever current.
2. Inspect worktree state and preserve unrelated changes. Do not start parallel work on the inline `webhook.py` document. Do not independently extract its HTML/CSS/JavaScript into new modules or choose a frontend stack.
3. Confirm #15 exists: `ChatStore`, the migration, conversation CRUD, WebSocket `conversation_id` and basic history restoration. It is already completed in this checkout. If working elsewhere without it, integrate that external prerequisite first; do not reimplement it from this package.
4. Consume the predecessor's evidence and DOM/state hooks before changing them. A documented proposed change is not evidence it is already implemented.

### Strict execution order and ownership

| Order | Specification / agent responsibility | Allowed implementation surface | Required handoff |
| --- | --- | --- | --- |
| 1 | [00 overview](00-overview-and-dependencies.md): establish baseline | Read only | Revision, current #15 presence, unresolved decisions and environment limits |
| 2 | [05 backend](05-backend-contracts-and-migration-boundaries.md): contract verification and bounded adjustments | Relevant Python handlers/helpers in `webhook.py`; metadata helper in `common/.../paperless.py`; focused backend tests assigned by 06 | Actual status/field behavior, added-date compatibility, availability behavior, focused test results; no HTML redesign |
| 3 | [01 shell](01-conversation-shell-and-responsive-layout.md): semantic layout and tokens | `chat_ui` markup/styles and only drawer/help/focus hooks | Preserved IDs, actual new hooks, breakpoints and keyboard evidence; existing interactions remain wired |
| 4 | [02 history](02-conversation-history-and-lifecycle.md): sidebar/lifecycle | Inline history/API/lifecycle functions and corresponding sidebar states | Selected-ID, readiness/loading/busy interface; request race/draft rules; CRUD/browser evidence |
| 5 | [03 turns](03-turn-rendering-tool-progress-and-composer.md): socket/composer/rendering | Inline socket listeners, turn/tool/Markdown/usage functions and composer states | Turn routing/busy/connection hooks, reconnect policy, shared live/restored renderer contract and evidence |
| 6 | [04 inspector](04-document-inspector-and-source-cards.md): source/preview | Inline source render/selection/reset and inspector markup/styles | Source selection/reset hooks, canonical navigation, unavailable behavior and evidence |
| 7 | [06 verification](06-test-and-verification-plan.md): integrated acceptance | Focused missing tests and necessary corrections within the responsible spec's scope | Test results, browser screenshots/scenarios, revision, skips and blockers |
| 8 | [07 final handoff](07-luna-agent-handoff.md): review checklist | Review/report only, unless a bounded defect is sent back to its owner | Final file manifest, completed order, evidence and outstanding decisions |

The filename order is a manifest, not a reason to postpone the backend contract check. Reading references from 05 to the UI specs creates no implementation cycle: those documents define consumer requirements; 05 delivers first. Verification agents may send defects back to the owning step, then rerun only the checks invalidated by the fix.

### Shared boundaries to hand off explicitly

- **History → turns:** one selected durable ID; history-loaded/lifecycle-loading flags; a single busy-state setter/derived enabled-state function; draft-discard behavior; safe transcript replacement. Names can follow existing code, but the handoff must state the actual names and locations.
- **Turns → inspector:** existing `renderSources(turnId, items)` hook (or an explicitly documented compatible replacement), stable turn/message keys, and an inspector reset hook called on successful history replacement/new/delete.
- **Shell → all:** preserved element IDs, actual drawer open/close hooks, focus restoration responsibilities and CSS token names. Avoid two independent focus managers or different breakpoint definitions.
- **Backend → consumers:** only the changes allowed by [05](05-backend-contracts-and-migration-boundaries.md). CRUD contract belongs to [02](02-conversation-history-and-lifecycle.md), event contract to [03](03-turn-rendering-tool-progress-and-composer.md), source shape to [04](04-document-inspector-and-source-cards.md); link those definitions rather than inventing another copy.

### Exit checklist for each assignment

- Assigned acceptance criteria are met, or an exact blocker is recorded with the unaffected work completed.
- Changed files/symbols and any departure from the spec are stated. Unrequested refactors, migrations and dependency changes are absent.
- Existing prerequisite behavior still works; no agent overwrote another's changes by restoring the baseline inline page.
- Applicable format/focused tests and browser scenarios were run; command, working directory, revision, result and skips are recorded.
- Success is not claimed for mocked-only database/proxy/Paperless integration; distinguish deterministic unit/browser fixture evidence from live service evidence.
- The next agent receives actual DOM/state/function hook names and any transient compatibility stub. Test-only mocks/stubs never ship as production history, archive metrics or document data.
- Changes remain reviewable; no commit, branch, PR publication or deployment occurs merely because an agent finished this package's assignment without separate authorization.

### Unresolved detail handling

The decision register in [00](00-overview-and-dependencies.md) is authoritative. First-delivery fallbacks are binding: omit unsupported account/upload controls, use real health rather than index counts, show provenance rather than relevance scores, and expose no generation-cancel control. Do not repeatedly ask the user to choose those already specified defaults.

Remaining verification details are: whether the deployed Paperless version supplies `added`; whether the actual proxy strips/sets the owner header for HTTP and WebSocket; and whether a same-origin authenticated browser test environment is available. The deployment owner must also confirm the shared-document access assumption before describing the system as enforcing per-user document ACLs. Record evidence when available; do not claim a schema migration or fabricated browser identity solves these questions.

Unresolved product extensions—real cancellation, upload semantics, profile actions, authoritative indexing metrics, calibrated relevance or historical model memory—require a separate decision before expansion. Their absence does not block the defined first-delivery visual scope. If a user subsequently requires one, update the owning contract/spec before independent agents implement it.

### Final review and report

Confirm the implementation preserves all behavior in [00](00-overview-and-dependencies.md), meets [06](06-test-and-verification-plan.md), and leaves #15's storage internals unchanged. Report any deployment/security verification limitation plainly. Do not call a manual screenshot comparison an automated test or claim execution by a model/subagent that was not actually used.

For the current documentation-only delivery, return the eight-file manifest in numeric filename order and the execution order below; attach unresolved details as brief file annotations when needed. Do not append application code or pseudocode. Later implementation handoffs additionally include changed product files and actual verification evidence.

## Acceptance criteria

- Every implementation file has one responsible sequential agent; shared-file edits do not overlap.
- #15 is recognized as implemented and consumed; no persistence task is duplicated.
- Each exit handoff gives the next agent concrete interfaces and evidence rather than just “done”.
- Final acceptance distinguishes passed checks, mocks, skipped infrastructure and unresolved product extensions.
- The planning delivery consists only of the eight requested Markdown files, with no commit or implementation artifacts.

## Verification

Before each assignment, from the repository root run `git status --short --untracked-files=all` and `git rev-parse HEAD`; inspect relevant predecessor changes with `git diff`. Follow the assigned spec's focused checks and [06](06-test-and-verification-plan.md). For final documentation scope, use `git diff --check` and verify the manifest contains exactly eight Markdown files and each has all eight required sections.

## Non-goals

No parallel overlapping edits, new project management system, ninth spec/README, independent database implementation, unrequested commits/deployment or work on unrelated repository issues.

## Dependencies

- Task/baseline: [00](00-overview-and-dependencies.md).
- Implementation sequence: [05](05-backend-contracts-and-migration-boundaries.md) → [01](01-conversation-shell-and-responsive-layout.md) → [02](02-conversation-history-and-lifecycle.md) → [03](03-turn-rendering-tool-progress-and-composer.md) → [04](04-document-inspector-and-source-cards.md).
- Final evidence: [06](06-test-and-verification-plan.md), then this checklist.

## File manifest and dependency order

1. `00-overview-and-dependencies.md`
2. `01-conversation-shell-and-responsive-layout.md`
3. `02-conversation-history-and-lifecycle.md`
4. `03-turn-rendering-tool-progress-and-composer.md`
5. `04-document-inspector-and-source-cards.md`
6. `05-backend-contracts-and-migration-boundaries.md`
7. `06-test-and-verification-plan.md`
8. `07-luna-agent-handoff.md`

Execution: **00 → 05 → 01 → 02 → 03 → 04 → 06 → 07**. Read 07 before assigning work. #15 is an implemented external prerequisite for 05 and all history integration.
