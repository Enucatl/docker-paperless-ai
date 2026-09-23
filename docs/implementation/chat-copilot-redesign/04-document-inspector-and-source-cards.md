# Document inspector and source cards

## Purpose

Turn the existing inline preview into the prototype's selected-document inspector while keeping Paperless as the source of current metadata, thumbnails, previews and full document navigation.

## In scope

Compact visible source cards, selected-source state, inspector metadata, iframe preview, empty/loading/unavailable states, responsive source selection and safe navigation. This file owns source presentation and its consumer data requirements.

## Existing code to inspect

- `ai/src/paperless_ai/search/webhook.py`: `renderSources`, `renderSourceBadges`, `openPreview`, `preview-*` elements, `_build_chat_sources`, `_restore_chat_sources`, `_document_detail_url`, `_document_thumb_url`, `_document_preview_url`.
- `common/src/paperless_common/paperless.py`: `get_document_for_chat`, `get_document_chat_metadata`, name resolvers and `get_tag_names`.
- `ai/src/paperless_ai/search/chat_store.py`: `load_conversation` source snapshots and `append_turn` source references, read only.
- `ai/tests/test_paperless_client.py`: metadata resolver tests; `ai/tests/test_chat_tools.py`: source flags; `ai/tests/test_chat_history.py`: persisted references.

## Implementation instructions

### Source data and freshness

| Field | Current contract / use |
| --- | --- |
| `id`, `title` | Integer Paperless document ID; title string, fallback `Document {id}` for absent snapshot data |
| `created` | Paperless document date; do not treat as import/add date |
| `correspondent_name`, `document_type_name`, `storage_path_name` | Nullable resolved strings |
| `tag_names` | Array of human-readable tag strings |
| `archive_serial_number`, `original_filename` | Optional metadata; preserve ID/ASN/storage path access even if shown only in inspector |
| `matched`, `inspected` | Boolean provenance flags, not relevance scores |
| `detail_url` | `/documents/{id}/detail` |
| `thumb_url` | `/api/documents/{id}/thumb/` |
| `preview_url` | `/api/documents/{id}/preview/` |
| `available` | Historical load supplies a boolean; existing successful live sources omit it, so absence is treated as available for compatibility |
| `relevance` | Nullable/untyped persisted JSON; live source builder does not supply a calibrated score. Do not interpret it as a confidence percentage or High relevance label. |
| `added` | Proposed optional live metadata addition owned by [05](05-backend-contracts-and-migration-boundaries.md), not an existing field |

Live sources are assembled after the agent returns. Reopened history refreshes metadata in `_restore_chat_sources`; display snapshots only when a live document cannot be obtained. Name resolver caches may delay label changes until refreshed; do not promise instant cross-device synchronization. Selecting a card uses the already returned metadata and loads a current thumbnail/preview URL; it does not introduce a separate metadata endpoint. An explicit history refresh while idle obtains metadata again.

A source set means documents matched and/or inspected by tools; it is not necessarily an exhaustive set of answer citations. Keep the returned order: live sources are inspected-first then ID, restored sources currently load by ID. Do not impose an inferred relevance sort or treat these differing orders as corruption.

### Cards and selected state

1. Render source cards visibly below the answer, introduced by “Sources (N)” using actual unique returned items. Do not hide all cards behind the current closed `details`. For many sources allow normal wrapping/vertical transcript scrolling; do not silently drop items or add pagination.
2. At the reference width, three approximately 220px cards fit across the answer, with 12px gaps. Use a minimum near 210px and reduce to two/one columns as space requires; below 768px use one column. Thumbnail is approximately 56 × 76px, object-fit contain, beside title/date/correspondent, with compact tag chips and provenance below. Card padding is 10px, corner radius 8px, normal 1px border and green selected border.
3. Show at most two tag chips in the compact card plus a plain “+N more” indicator; the inspector lists all tags. Show actual Matched and Read in full text with distinguishable icons when true. Do not substitute the screenshot's unsupported relevance labels.
4. A dedicated title/thumbnail selection control and primary “Preview document” button select the source in-app; keep “Open in Paperless” as a separate full-document link. This resolves the prototype's ambiguous “Open document” action while preserving both existing capabilities. Do not nest links/buttons inside another button or make an article keyboard-clickable without semantics.
5. Store the selected source in memory as conversation ID, turn/message key and document ID; repeated references in different answers select the originating card. Reflect selection with a visible border and an accessible pressed/current indication on the preview control. Selection is presentation state, never durable data.
6. Start with an empty inspector. Do not auto-select the first source on answer completion or history load. On explicit selection, show the chosen document and open the responsive drawer when required. Desktop selection keeps focus on the card; modal focus behavior follows [01](01-conversation-shell-and-responsive-layout.md).
7. Closing clears the selection and iframe source, releases the desktop inspector column, and restores focus. Source cards remain usable to reopen it. New/load/delete clears selection before replacing transcript content. Ordinary sidebar refresh after turn completion does not clear an existing selection.

### Inspector content

Show title, all tags, “Open in Paperless” and Close at the top. Use a definition list for Correspondent, Document date, Document type, Tags, Added and File name, matching the reference hierarchy. Also retain Document ID, and show ASN/storage path when available. Format real dates with the browser locale; date-only values must not shift a day due to timezone conversion. Invalid/absent fields show “Not available”; do not replace the added date with conversation timestamps.

Tags may appear as header chips and in the Tags row, as in the image, but there are no tag-filter actions in this scope. Escape all labels via text nodes. Long filenames and correspondents wrap. Missing tags show “No tags”.

Use the existing Paperless `preview_url` in a titled iframe. Retain browser/Paperless-native PDF toolbar behavior, including search/zoom/pages if provided; do not draw nonfunctional lookalike controls. Keep full-document navigation available even when embedding is unsupported. Avoid fetching/storing PDF bytes, OCR or thumbnails in the AI history database or browser storage.

Set a visible “Loading preview…” status until the frame load event. A load event alone does not prove a PDF succeeded: an authentication/error HTML page can also load. Handle observable frame/network errors, and keep a persistent “Preview not displaying? Open in Paperless” fallback without claiming complete error detection. Do not inspect embedded PDF internals or read/copy protected content to implement controls. Switching sources must immediately update title/metadata and prevent an earlier frame callback from clearing the newer document's loading state.

### Unavailable and safe-navigation behavior

For `available: false`, keep the historical title/ID with “Document unavailable” and make clear metadata may be a saved snapshot. Disable thumbnail/preview loading, present an explanatory inspector state if selected for details, and do not leave the preceding document visible. A disabled preview button is separate from a details-selection control so the unavailable reference remains inspectable. A broken thumbnail on an otherwise available source uses a document placeholder, not a broken-image glyph, and does not disable its full-document link.

Retain the known “Open in Paperless” URL for an unavailable reference as a recovery path, labelled that Paperless may require access or report it missing. Never interpret 403 as proof of deletion. Per-source metadata failure isolation is owned by [05](05-backend-contracts-and-migration-boundaries.md); a single inaccessible source must not erase the rest of a restored transcript.

Only accept the canonical same-origin document paths associated with that source ID for source actions. Historical metadata/URLs and model links are untrusted data. Do not assign arbitrary persisted/model URLs to an iframe or allow `javascript:`/external preview destinations. Full-document links open in a new tab with `noopener noreferrer`. Ordinary sanitized Markdown links keep normal link behavior; do not intercept every numeric mention or arbitrary URL as a source selection.

Paperless browser session and AI-service token access are distinct; the AI's metadata visibility does not guarantee the browser can view a PDF. Preserve that boundary and follow the deployment verification in [05](05-backend-contracts-and-migration-boundaries.md); do not proxy documents with the service credential to bypass browser authorization.

## Acceptance criteria

- Live and restored sources use the same cards and selection behavior; repeated references do not select the wrong turn.
- Selected card has a green outline; inspector displays real metadata and the matching preview; clearing/changing history never leaves stale document content.
- Existing root-relative thumbnail/preview/detail paths work under `/ai/chat` and never receive the AI prefix.
- Missing metadata, no tags, long names, broken thumbnail, unavailable document and embedding/authentication failures have readable fallbacks.
- No arbitrary persisted URL can become an iframe destination; Markdown and source labels remain safe.
- Added date is sourced only from its explicit live field; relevance remains factual provenance rather than invented confidence.
- Full Paperless navigation remains available on desktop and mobile independently of preview success.

## Verification

After implementation, from `ai/` run `uv run ruff format .` and `uv run pytest tests/test_paperless_client.py tests/test_chat_tools.py -q`. Backend metadata changes belong to [05](05-backend-contracts-and-migration-boundaries.md). Run persistence/live-document checks through `./run_tests.sh` per [06](06-test-and-verification-plan.md). In a browser select A then B rapidly, close/reopen, switch conversations, revoke/delete a test document, expire the Paperless session, and check mobile focus and native PDF behavior. Do not claim iframe load proves document availability.

## Non-goals

No custom PDF viewer, downloads proxy, document mutation, tagging, uploads, OCR persistence, relevance calculation, new document datastore, automatic first-source selection or citation parsing heuristics.

## Dependencies

- Start after shell [01](01-conversation-shell-and-responsive-layout.md), history [02](02-conversation-history-and-lifecycle.md), turn hooks [03](03-turn-rendering-tool-progress-and-composer.md) and backend adjustments [05](05-backend-contracts-and-migration-boundaries.md).
- External #15 source-reference contract is implemented; preserve it per [00](00-overview-and-dependencies.md).
- Integration and visual gates: [06](06-test-and-verification-plan.md).
