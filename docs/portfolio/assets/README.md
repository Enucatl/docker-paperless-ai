# Portfolio Media Asset Checklist

Add redacted real media to this directory when you are ready to publish the
portfolio write-up. Keep personal document contents, names, addresses, account
numbers, tokens, hostnames, and internal URLs out of the final images.

## Required Assets

### `chat-demo.gif` or `chat-demo.webm`

Capture the `/chat` interface answering a realistic question about personal
documents.

Recommended sequence:

1. Open the chat UI.
2. Ask a query that needs search, source inspection, or metadata filtering.
3. Show the visible tool-call panel while the agent searches or reads a source.
4. Show the final answer with source cards visible.

Redact:

- document titles if they reveal private information,
- names, addresses, invoice numbers, claim numbers, account numbers,
- local domains, IP addresses, API keys, and tokens.

### `architecture-overview.png`

Create a diagram that shows:

- Paperless-ngx document import and workflows,
- the thin webhook listener,
- Redis queues and stage tags,
- the AI worker stages: OCR, metadata extraction, embedding,
- model providers or local vLLM endpoints,
- Qdrant for chunk vectors,
- the browser chat UI and `/search` endpoint,
- Phoenix for traces and evaluation results.

Use service names rather than private hostnames.

### `phoenix-trace.png`

Capture a redacted Phoenix trace for one chat or pipeline run.

It should make clear that the project records spans for:

- chat turn execution,
- tool calls,
- retrieval steps,
- LLM calls and token/cost metadata where available.

### `eval-comparison.png`

Capture a redacted Phoenix experiment comparison or terminal comparison table.
The useful story is that OCR and metadata model choices were evaluated against a
golden dataset instead of picked by intuition.

## Optional Asset

### `pipeline-tags.png`

Capture a Paperless document or workflow view showing the tag-based stage
transition, such as `ai:run-ocr`, `ai:run-metadata`, or `ai:run-embed`.

Only include this if the visual helps. The architecture diagram and chat demo
are more important.
