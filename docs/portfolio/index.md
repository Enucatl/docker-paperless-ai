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
- [Media asset checklist](assets/README.md) lists the screenshots and videos
  that should be added manually after redaction.

## Showcase Media To Add

The pages intentionally include placeholders instead of checked-in demo media.
Add redacted real screenshots or videos under `docs/portfolio/assets/` using
the filenames below:

| File | Purpose |
|---|---|
| `chat-demo.gif` or `chat-demo.webm` | Redacted chat UI answering a realistic document question |
| `architecture-overview.png` | System diagram from Paperless through Redis, workers, Qdrant, models, and Phoenix |
| `phoenix-trace.png` | Redacted trace view showing model/tool spans and token/cost metadata |
| `eval-comparison.png` | Redacted evaluation comparison across model configurations |
| `pipeline-tags.png` | Optional Paperless tag/workflow screenshot for the staged ingestion pipeline |

The placeholders are written so the docs still render before the files exist.
