# Paperless AI Portfolio Notes

This section explains the project as a portfolio case study instead of a setup
manual. The root [README](../../README.md) stays focused on running and
operating the stack; these pages focus on the product problem, model choices,
retrieval design, evaluation workflow, and production engineering tradeoffs.

## Start Here

- [Case study: AI document copilot for paperless-ngx](case-study.md) is the
  hiring-manager overview. It emphasizes the problem framing, architecture,
  product outcome, and engineering maturity.
- [Technical deep dive: ingestion, RAG, and observability](rag-deep-dive.md)
  is the ML engineering write-up. It goes deeper on retrieval, agent tools,
  evaluation, local model tradeoffs, and Phoenix observability.
- [Media assets](assets/README.md) documents the screenshots, videos, and
  diagram sources used by these pages.

## Showcase Media

The portfolio pages use these local assets:

| File | Purpose |
|---|---|
| `chat-demo.webm` | Redacted chat UI video for the tax final bills query |
| `chat-demo.png` | Still image showing the trace detail for the chat demo |
| `data-ingestion-flow.png` | Data ingestion flow from Paperless workflows to OCR, metadata, embeddings, and Qdrant |
| `data-ingestion-flow.mmd` | Mermaid source for the data ingestion diagram |
| `agentic-chat-flow.png` | Copilot chat flow through tools, hybrid search, reranking, judging, and final response |
| `agentic-chat-flow.mmd` | Mermaid source for the agentic chat diagram |
| `phoenix-trace.png` | More complex Phoenix chat trace with a longer agentic tool flow |
| `eval-comparison.png` | Evaluation comparison across model configurations |
| `full-metadata-trace.png` | Phoenix trace/cost view for the selected local OCR plus Gemini metadata setup |
