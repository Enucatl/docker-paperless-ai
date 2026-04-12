# Media Assets

This directory contains the media used by the portfolio write-up. Assets are
redacted so personal document contents, names, addresses, account numbers,
tokens, hostnames, and internal URLs are not exposed.

## Assets

### `chat-demo.webm`

Video of the `/chat` interface for the tax final bills query. It shows the user
query, the visible tool-call panel, the final answer, and source cards.

### `chat-demo.png`

Still image accompanying the chat demo. It shows the trace detail for the same
turn, including the model/tool stack and cost context.

### `data-ingestion-flow.png`

Exported architecture diagram generated from
`data-ingestion-flow.mmd`. It shows:

- Paperless-ngx document import and workflows,
- the thin webhook listener,
- Redis queues and stage tags,
- the AI worker stages: OCR, metadata extraction, embedding,
- model providers or local vLLM endpoints on the GPU workstation,
- Qdrant for chunk vectors,
- Phoenix for traces and token/cost visibility.

Regenerate after editing the Mermaid source:

```bash
npx --yes @mermaid-js/mermaid-cli \
  -p /tmp/mermaid-puppeteer-config.json \
  -i data-ingestion-flow.mmd \
  -o assets/data-ingestion-flow.png \
  -b white \
  -w 2000
```

### `agentic-chat-flow.png`

Exported agentic chat diagram generated from
`agentic-chat-flow.mmd`. It shows the user query entering the
LangGraph agent, tool fan-out, metadata and full-document reads through the
Paperless REST API, hybrid search over Paperless/Postgres and Qdrant, local
`bge-reranker-v2-m3` reranking, an LLM precision judge, and the final
source-backed response.

Regenerate after editing the Mermaid source:

```bash
npx --yes @mermaid-js/mermaid-cli \
  -p /tmp/mermaid-puppeteer-config.json \
  -i agentic-chat-flow.mmd \
  -o assets/agentic-chat-flow.png \
  -b white \
  -w 1400
```

### `phoenix-trace.png`

Advanced Phoenix trace for a more complex chat turn. It shows a longer agentic
flow with more tool calls than the tax final bills example, which makes it a
good second observability example after the simpler demo trace.

### `eval-comparison.png`

Phoenix experiment comparison for the OCR and metadata model matrix. The useful
story is that OCR and metadata model choices were evaluated against a golden
dataset instead of picked by intuition.

### `full-metadata-trace.png`

Phoenix trace/cost view for the chosen setup: local OCR, local BAAI/bge-m3
embeddings, and Gemini 3.1 flash-lite for metadata extraction. This supports the
cost claim from the real backfill: about 2,000 documents, roughly 7,000 pages,
and less than one dollar in Google API credits.
