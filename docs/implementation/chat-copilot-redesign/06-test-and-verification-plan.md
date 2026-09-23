# Test and verification plan

## Purpose

Define bounded evidence that the redesign matches the reference, preserves completed #15 and handles real interaction failures. This file specifies future checks; no tests, fixtures, screenshots or application changes are delivered with this planning package.

## In scope

Existing test inventory, focused additions for meaningful behavior, Docker integration, browser/manual acceptance, security/accessibility checks and concrete repository commands. Collect evidence once per changed behavior; avoid broad repeated runs without new failures or edits.

## Existing code to inspect

- `ai/tests/test_chat_tools.py`: deterministic agent/tool events, source flags, usage aggregation.
- `ai/tests/test_chat_history.py`: PostgreSQL append/reload, owner-isolated load and cascading deletion; skips without `CHAT_DATABASE_URL`.
- `ai/tests/test_paperless_client.py`: async client and cached metadata resolver tests; not yet a complete source-metadata/inspector suite.
- `ai/tests/test_webhook_startup.py`: Paperless initialization retries; not chat UI/HTTP/WebSocket coverage.
- `ai/tests/test_webhook.py`: webhook ingress/workflow tests, including `webhook_session`; this is not a copilot transport suite despite its filename.
- `ai/tests/test_search.py`: mixed local and infrastructure-dependent search tests.
- `ai/tests/conftest.py`: infrastructure skip hooks, `COPILOT_URL`, `paperless_client`, document lifecycle fixtures, inference mocks.
- `run_tests.sh`, `docker-compose.test.yml`, `ai/Dockerfile`, `ai/pyproject.toml`, `ai/uv.lock`.

## Implementation instructions

### Test ownership and minimum additions

The backend agent owns focused tests for its changed boundaries while implementing [05](05-backend-contracts-and-migration-boundaries.md). Later UI agents supply reproducible browser scenarios for their own acceptance criteria. The final verification agent closes gaps, rather than writing duplicate suites for each function or matching static markup strings.

Use the existing pytest/unittest mock patterns. A single new `ai/tests/test_chat_ui_contracts.py` is an appropriate bounded location for transport/source boundary checks; it does not exist at the baseline. Keep persistence integration in the existing `test_chat_history.py` and metadata helper coverage in `test_paperless_client.py`. These are future implementation assignments, not files to create in this documentation task.

| Area | Meaningful required evidence | Owner |
| --- | --- | --- |
| CRUD validation and ownership | Valid create/list/load/rename/delete shapes; default title; 120-character/blank/null titles; invalid UUID vs unknown UUID; foreign owner cannot list/load/mutate | Backend, preserving #15 store integration |
| Completed turn | Deterministic socket harness records start/tool events, append before final answer, one success completion; save failure yields no final successful answer/sources; deleted record fails append | Backend; retain existing agent test |
| Legacy/sentinel behavior | Plain-text and omitted-ID fallback still work; invalid/unavailable errors need no ordinary started turn; malformed IDs do not hit store | Backend |
| Source refresh | Current metadata overrides snapshot; added nullable; missing/403/temporary failure preserves unavailable reference; successful peers survive; no OCR/PDF bytes copied | Backend/shared helper |
| Socket/UI state | Double send, Enter/IME, loading history with open socket, active-turn navigation, late events and non-JSON message do not corrupt selected transcript | Turn agent, browser |
| Lifecycle | Slow A/B load ordering, double New, draft confirmation, failed delete/replacement, rename error/retry, reload and second-device restoration | History agent, browser plus Docker integration |
| Safe rendering | Malicious Markdown/link schemes and source/tool strings cannot execute; marked/DOMPurify failure falls back to text; canonical preview paths | Turn/inspector agents, browser |
| Source selection | Repeated source in two turns, rapid A/B preview, close/reset, deleted/restricted document, broken thumbnail and PDF auth failure | Inspector agent, browser |

For server event tests, invoke the application boundary with controlled fake/AsyncMock socket and store/agent dependencies, or use an already available compatible client. Do not add an HTTP-client dependency merely because a copied FastAPI TestClient example expects one; this project's declared test extras do not explicitly include httpx. Do not call real inference to test transport state.

The autouse `mock_litellm` fixture patches several inference imports, but `chat_agent.py` holds its own imported `complete`; follow `test_chat_tools.py` and patch `paperless_ai.search.chat_agent.complete` explicitly. Tests inside `ai-test` also cannot patch the separately running `ai` service. Use deterministic in-process transport tests for model success/error sequences; use the running container for real CRUD/metadata/infrastructure checks. A dummy key is not a working model mock.

### Repository commands

Commands here are verification instructions, not pseudocode or application implementations.

For environment setup and focused existing local regression checks, working directory `ai/`:

```bash
uv sync --extra test --extra eval
uv run pytest tests/test_chat_tools.py tests/test_paperless_client.py tests/test_webhook_startup.py -q
```

After the proposed focused boundary file exists, still from `ai/`:

```bash
uv run pytest tests/test_chat_ui_contracts.py -q
```

After Python edits, run `uv run ruff format .` from each affected Python project (`ai/`, and `common/` if its helper changed). Run common consumer tests from `ai/`. From repository root run `git diff --check`. Inspect formatter output/diff so unrelated user changes are not swept into the task.

For a broader local regression check only when changes justify it, from `ai/`:

```bash
uv run pytest tests/ -k "not test_webhook and not test_phase_b_pipeline and not test_search"
```

This repository-prescribed selection is not proof that every collected case is infrastructure-free; `test_chat_history.py` may skip without a database. Report passed/skipped separately. Never point a local convenience test at the production conversation database.

For checks needing PostgreSQL, Paperless, Redis, webhook delivery or container networking, from repository root:

```bash
./run_tests.sh
```

Only reuse an already built test image when it contains the exact current code/tests:

```bash
./run_tests.sh --no-build
```

`run_tests.sh` accepts only `--no-build`; it does not accept pytest paths or `-k`. Do not document invented filtering flags. It builds `ai`, `ai-test` and `webhook-listener`, uses project `paperless-ai-test`, starts dependencies, runs the suite in `ai-test`, then tears down test containers/anonymous volumes with a trap. PostgreSQL and Redis use ephemeral storage. The test override deliberately uses its ephemeral Paperless database/test role for chat, not production's least-privilege `ai_chat` role; a pass therefore does not prove production role isolation.

The harness sets `MANAGE_PAPERLESS_WORKFLOWS=false`. Any webhook checks added incidentally must use `webhook_session` except intentional missing/invalid-token cases and must clean up workflows/documents. There is no need to expand webhook testing for a UI-only behavior.

### Browser setup and limitations

The repository has no `package.json`, frontend build, Playwright/Cypress suite or browser-test command. Do not claim one exists, add a JavaScript stack just for this redesign, or put invented `npm test` commands in handoffs. Use available browser automation tooling if the implementation environment provides it; otherwise perform and record the manual matrix below. Automated response/socket fixtures must conform to [02](02-conversation-history-and-lifecycle.md) and [03](03-turn-rendering-tool-progress-and-composer.md), and remain test-only.

Use an authorized disposable conversation/document in the real same-origin `/ai/chat` environment to verify routing and Paperless assets. `ai-test`'s internal `COPILOT_URL` is `http://ai:8001`, and the existing Docker harness does not expose a complete browser-facing Paperless/AI reverse proxy. Its cleanup also removes the environment on exit. Direct `/chat` can verify path derivation and markup but cannot establish same-origin Paperless asset/auth behavior by itself. If a browser-accessible test proxy is unavailable, mark those integration checks unverified and record the exact missing environment rather than modifying production routing.

For deterministic visual inspection, use test data resembling the reference: a long user question; metadata/search/read activity; an answer with a three-row table; three source cards; one selected document with tags, correspondent, dates and filename. Keep this data in the browser test/mocking environment, not production HTML. Do not copy the prototype's person's name or tax amounts into production fixtures by default.

### Manual visual and interaction checklist

| Scenario | Required observation |
| --- | --- |
| 1672 × 941 reference view | Green approximately 52px bar, approximately 300px sidebar, open right inspector, pale center, heading, visible activity, answer/table, three compact source cards, composer near bottom |
| 1440 / 1280 / 1024 / 768 widths | Breakpoint transitions match [01](01-conversation-shell-and-responsive-layout.md); no clipped center or hidden actions; drawers restore focus |
| 390 × 844 and 320px width | Single-column cards, full-screen inspector, reachable New/send/close, no page-wide horizontal scrolling |
| 200% zoom, short viewport, mobile keyboard | Text/controls remain reachable; composer does not cover transcript/inputs; tables scroll locally |
| Keyboard/reduced motion | Visible focus, skip link, labelled icons, modal Escape/focus restoration, no focus behind drawers, useful non-color states and reduced animation |
| Turn states | Pending, running, final, failed and unavailable service are distinct; no stuck spinner or duplicate sends |
| History states | Loading, empty, one/many entries, long titles, date groups, rename/delete confirmation and recoverable API errors |
| Reconnect/save race | Draft/recovery text survives, history reloads, no auto-resend, no cancellation claim, context-reset notice visible |
| Inspector states | Empty, selected, loading, unavailable, thumbnail failure, unsupported/unauthenticated preview; correct full-document link always reachable |
| Content safety | Malicious tool/source/Markdown content inert; missing Markdown libraries still display text |
| Truthful prototype adaptations | No invented profile/count/index time/relevance; no inert attachment or cancel control |

Record screenshots at reference desktop and mobile widths outside production assets, plus browser/version, viewport, revision, scenario and limitations in the implementation handoff. Native PDF toolbar appearance may differ by browser and is not a pixel-match failure. Color/spacing differences should be assessed on the page surrounding it.

### Release gate and evidence record

Require focused tests for changed logic, successful isolated persistence integration, browser interaction evidence, and the visual checklist before declaring the implementation accepted. Verify proxy identity-header handling and Paperless session behavior in the deployment environment; local mocks cannot prove them. Report infrastructure skips, unavailable browser/proxy checks and unresolved product extensions separately from passes.

Record each result as command/scenario, working directory, tested revision, result and blocker if any. Stop broadening tests when these gates pass unless later edits invalidate them. This specification package itself needs only the documentation checks in [00](00-overview-and-dependencies.md).

## Acceptance criteria

- Test claims identify actual existing coverage and future additions; a skipped database test is not called a pass.
- Future changed validation/source behavior has focused automated checks; UI behavior has reproducible browser evidence.
- Both proxied `/ai/chat` routing and restored/live Paperless source flows are checked where the required environment exists.
- No real inference expense or production database mutation is required for the regression suite.
- Desktop/mobile visual, keyboard, error and reconnect cases have explicit pass/fail evidence.

## Verification

The commands and matrix above are the verification procedure for implementation agents. For this document, verify every existing path/command against the baseline and every proposed test path is labelled future work. Confirm `run_tests.sh` options and Docker cleanup semantics by reading the script, not by launching services during documentation authoring.

## Non-goals

No tests or executable fixtures in the current delivery; no new browser framework, screenshot infrastructure project, benchmark, exhaustive unrelated regression expansion, production data cleanup or unrequested deployment.

## Dependencies

- Requirements come from [01](01-conversation-shell-and-responsive-layout.md), [02](02-conversation-history-and-lifecycle.md), [03](03-turn-rendering-tool-progress-and-composer.md), [04](04-document-inspector-and-source-cards.md) and [05](05-backend-contracts-and-migration-boundaries.md).
- Implemented #15 boundary and capability decisions: [00](00-overview-and-dependencies.md).
- Report evidence through [07](07-luna-agent-handoff.md).
