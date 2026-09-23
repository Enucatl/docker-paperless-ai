# Conversation shell and responsive layout

## Purpose

Translate the inspected [prototype](https://raw.githubusercontent.com/Enucatl/docker-paperless-ai/design/chat-copilot-prototype/docs/assets/chat-copilot-prototype.png) into the existing inline page. Establish the shared visual/semantic structure before behavior agents work on it.

## In scope

Shell DOM, local design tokens, three-column desktop geometry, smaller-screen drawers, scrolling, focus and keyboard access. Establish placeholders/hooks for history, turns, composer, archive status and inspector without reimplementing their business behavior.

## Existing code to inspect

- `ai/src/paperless_ai/search/webhook.py`: all of `chat_ui()`, especially `.paperless-topbar`, `.chat-layout`, `.history-panel`, `.conversation`, `.composer`, `.preview-panel` and the 1080/720px media rules.
- Same file: element bindings at the beginning of the script and `autosizePrompt`, `openPreview`, `createTurn`, `renderConversationList`.
- `docker-compose.yml`: `paperless-ai-strip` and Paperless static-asset routing.
- [00](00-overview-and-dependencies.md): baseline and prototype capability decisions.

## Implementation instructions

### Visual evidence and tokens

The reference is 1672 × 941 pixels. Its bar is approximately 51px tall; sidebar ends near x=300, inspector begins near x=1157, and center heading begins near x=320. The composer is inset approximately 20px from center edges, close to the viewport bottom. Treat measurements/colors below as deliberate implementation targets inferred from the image, not extracted source CSS. Keep font sizes in rem relative to a 16px default.

| Token / element | Target |
| --- | --- |
| Bar | 52px minimum height, deep green `#17541f`, white text; 20px horizontal padding |
| Canvas / panels | `#f8faf9` center/sidebar, `#ffffff` answer/inspector/composer |
| Text / muted / border | `#111827`, `#64748b`, `#dce3e8` respectively; adjust muted text if contrast testing requires it |
| Active history / user bubble | Soft green `#e4f0e6`; selected-source border `#18752b` |
| Font | Existing Paperless Bootstrap/system sans-serif stack; no external font |
| Typography | Page heading 34px/1.2, weight 700; inspector title 20px/1.3, weight 600; body/answer 16px/1.5; dense metadata/history 14px/1.4; secondary timestamps 12px/1.4 |
| Spacing | 4, 8, 12, 16, 20, 24, 32px scale; center inset 20px; inspector inset 20px; sidebar inset 12–18px |
| Borders / corners | 1px panel dividers; 8px cards; 12px answer/composer; pill-shaped prompt input; very subtle preview shadow only |
| Controls | New chat 46px tall, full sidebar width; send 44px round; all compact icon controls at least 44 × 44px hit area |
| User/assistant markers | Approximately 36px circles; generic user and assistant icons, no invented initials |
| Source cards | Compact thumbnail and text row, tags/status below, full-width primary preview action; detailed sizing belongs to [04](04-document-inspector-and-source-cards.md) |

The composition should be flush with the viewport, with thin vertical separators. Replace the current 1380px maximum-width shell and floating bordered column panels. Use three columns at wide sizes, not a centered dashboard of separate cards. Keep white answer and composer surfaces visually distinct from the pale canvas.

### Semantic structure and hooks

1. Use a page header containing the Paperless brand link, divider, “AI Copilot”, socket status with text, and Help. Link the brand to Paperless `/`. Use an existing available brand asset if verified; otherwise a simple document icon and text suffice. Help opens a small accessible explanation of keyboard shortcuts and the documented context limitation. Do not add account/attachment placeholders; follow [00](00-overview-and-dependencies.md).
2. Provide a skip link to the center main region. The history is labelled navigation with “New chat”, grouped conversation buttons, active-conversation actions and archive footer. Preserve `history-list`, `new-conversation`, `rename-conversation`, `delete-conversation` IDs.
3. The main region has the single page heading “Ask your archive”, subtitle “Search, inspect, and understand your Paperless documents.”, an error/status area, scrollable transcript, and composer. Preserve `conversation`, `socket-banner`, `chat-form`, `prompt`, `send-button`. Use labelled turn articles, with semantic time elements only when a real timestamp is available.
4. The inspector is a labelled complementary region containing selected title, “Open in Paperless”, close button, metadata definition list, and the existing iframe. Preserve the `preview-*` IDs. Empty inspector text explains how to select a source.
5. Introduce only the few hooks needed to open/close history and inspector at compact widths. Keep logical DOM order header → history → main → inspector. Avoid duplicate copies of history/inspector for different breakpoints.

### Layout rules

| Viewport width | Required geometry and behavior |
| --- | --- |
| At least 1440px | Persistent 300px history, flexible center with a 560px minimum, inspector at 31% of viewport clamped to 440–520px. At 1672px this approximates 300 / 854 / 518px. Closed inspector releases its space to center. |
| 1200–1439px | Persistent 240px history and flexible center; inspector becomes a right modal drawer, width 480px capped at viewport width. Selecting a source opens it. |
| 768–1199px | Center occupies width; history opens as a left modal drawer, 300px wide; inspector remains a right drawer, 480px maximum. Only one drawer open at a time. |
| Below 768px | Header allows brand text to shorten without hiding status; 12px center insets, 26px heading. History is at most 320px and at most viewport minus 32px. Inspector is full-width, with explicit Back/Close. Cards stack to one column. |

Use available dynamic viewport height with a fallback, and account for safe-area bottom padding. Header and composer remain visible while transcript scrolls. Sidebar list scrolls independently with its footer at the bottom. Inspector metadata/preview can scroll without pushing the composer offscreen. At short heights/large text, allow vertical scrolling rather than clipping controls. Test the virtual keyboard: it must not cover the prompt or send action.

Do not leave the old single-column-at-1080px rule active underneath the new layout. Tables and code blocks may scroll horizontally inside their answer; the page itself must not. Long titles, tags and filenames wrap or truncate with an accessible full value without expanding grid columns.

### Accessibility and keyboard navigation

- All icon controls have accessible names and a visible focus ring. Connection/progress/selection states include text or shape, not color alone. Normal text needs 4.5:1 contrast; focus and meaningful UI boundaries need 3:1.
- Keep normal Tab navigation through semantic buttons/links. Mark the selected history entry with `aria-current`; source selection uses the button semantics specified in [04](04-document-inspector-and-source-cards.md). Do not invent a listbox keyboard model.
- Modal drawers/dialogs move focus to their heading or first meaningful control, keep focus inside while open, close on Escape, and restore focus to the opener. Background controls are inert while a modal is open. A desktop inspector is non-modal and does not trap focus.
- Closing the desktop inspector returns focus to the selecting source control. Resize transitions must release modal/inert state and leave focus on a visible element. An unavailable opener falls back to the history toggle or transcript heading.
- Use a concise polite status region for connection/tool/completion announcements; use an alert for actionable failures. Do not repeatedly announce the entire Markdown answer or auto-focus it on completion.
- Honor reduced-motion preferences; disable activity animation when requested. Preserve reading position when the user has scrolled up, as defined in [03](03-turn-rendering-tool-progress-and-composer.md).

## Acceptance criteria

- At 1672 × 941, the header, three columns, heading, answer and bottom composer match the reference hierarchy and approximate proportions.
- At 1440, 1280, 1024, 768 and 390px widths, every control remains usable and the page has no horizontal overflow; at 320px/200% zoom, content remains reachable.
- Drawer close/Escape/resize restores usable focus; tabbing cannot enter hidden panes.
- Existing IDs and behavior hooks continue working; no framework or new asset pipeline is introduced.
- New chat, connection state and full Paperless navigation remain understandable without icons or color.

## Verification

After implementing this spec, from `ai/` run `uv run ruff format .` because the inline page is inside Python. From the repository root run `git diff --check`. Inspect the rendered `/ai/chat` in a browser at the widths above; test keyboard-only navigation, 200% zoom, reduced motion, long metadata and mobile keyboard behavior. Follow the browser environment instructions in [06](06-test-and-verification-plan.md); static HTML string assertions alone cannot verify layout.

## Non-goals

No persistence/backend changes, data fetching redesign, fabricated user profile, uploads, pixel-identical native PDF toolbar, dark theme, new icon library or broad extraction of the inline page into a frontend project. Do not implement the event or history policies owned by later specs.

## Dependencies

- Read [00](00-overview-and-dependencies.md) and the compatibility decisions in [05](05-backend-contracts-and-migration-boundaries.md) first.
- Provide shell hooks to [02](02-conversation-history-and-lifecycle.md), [03](03-turn-rendering-tool-progress-and-composer.md) and [04](04-document-inspector-and-source-cards.md).
- Verification and delivery gates: [06](06-test-and-verification-plan.md), [07](07-luna-agent-handoff.md).
