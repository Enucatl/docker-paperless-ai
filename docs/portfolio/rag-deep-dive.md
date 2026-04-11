# Technical Deep Dive: Ingestion, RAG, and Observability

This write-up focuses on the ML and retrieval architecture behind the
paperless-ngx AI layer. It complements the [case study](case-study.md), which
is the higher-level portfolio narrative.

## Ingestion Pipeline

The ingestion pipeline starts with Paperless workflows rather than a custom file
watcher. New or backfilled documents enter the AI pipeline by receiving stage
tags such as `ai:run-ocr`, and the webhook listener enqueues the document ID in
Redis.

> Optional media placeholder: add `assets/pipeline-tags.png` here if it helps.
> Show the stage tags or Paperless workflow state with document details redacted.

The pipeline stages are independent:

- OCR downloads the original PDF, renders pages, sends page images to the
  configured vision model, and writes the transcript back to Paperless content.
- Metadata extraction reads the transcript from Paperless, extracts structured
  fields, and patches Paperless through the REST API.
- Embedding reads the final content and metadata, chunks the document, embeds
  chunks, and upserts dense and sparse vectors into Qdrant.

The separation matters because the operational profile of each step is
different. OCR may need a vision-capable model and larger image payloads.
Metadata extraction is usually a smaller text-only call. Embedding is a batch
indexing task. Chat is interactive and must keep latency acceptable. The repo
therefore exposes independent model and API-base settings for OCR, metadata,
and chat instead of tying the whole system to one endpoint.

## Model Selection and Evaluation

The repository includes a Phoenix-based evaluation flow around a golden dataset
of scanned documents. Experiments are configured in
`ai/src/paperless_ai/eval/experiments.yaml` and can vary OCR models, metadata
models, API bases, temperatures, reasoning parameters, and judge models.

The evaluation exists because OCR and document metadata extraction produce
errors that are expensive to spot casually. Exact dates, null correspondents,
and document titles need different scoring behavior, so the eval suite reports
multiple metrics rather than a single pass/fail number.

Model families represented in the experiment configuration include:

- hosted Gemini variants for OCR, metadata extraction, and judging,
- Nanonets-OCR2-3B served through an OpenAI-compatible vLLM endpoint,
- local metadata extraction candidates such as NuExtract,
- Qwen configurations with thinking enabled or disabled and explicit sampling
  parameters.

The important engineering choice is that local and hosted models are not treated
as ideology. Local models can improve privacy and cost predictability, while
hosted models may win on quality, latency, or maintenance burden. The
architecture keeps those decisions reversible by making model endpoints
configuration, then uses evaluation to decide what is good enough for the
document workload.

> Media placeholder: add `assets/eval-comparison.png` here.
> Use a redacted Phoenix experiment table or terminal comparison from the eval
> runner.

## Retrieval Design

The search layer is hybrid. During indexing, chunks are written to Qdrant with
named dense and sparse vectors from bge-m3-compatible embeddings. At query time,
the shared retrieval pipeline combines:

- dense vector search against Qdrant,
- keyword search through the Paperless API,
- Reciprocal Rank Fusion over dense and keyword document rankings,
- local reranking of chunk candidates with `BAAI/bge-reranker-v2-m3`,
- document-level deduplication after chunk-level scoring.

The browser chat and `/search` endpoint share the retrieval implementation, but
chat can choose precision or recall behavior. Precision uses a smaller retrieval
surface and can apply an LLM judge to filter candidates. Recall increases the
dense candidate pool and requires the agent to provide an explicit limit for
broader list-style questions.

The local query embedder and reranker run in a process-backed worker. This keeps
the FastAPI process responsive while allowing the memory-heavy local models to
load lazily and exit after an idle timeout.

## Agentic Chat

The chat copilot is a LangGraph-based loop around a LiteLLM chat model. The
agent receives tool schemas and decides when to call them. The available tools
are intentionally narrow:

- `get_available_metadata` returns exact Paperless correspondent, document
  type, storage path, and tag names before filtered searches.
- `search_documents` runs hybrid retrieval with optional metadata filters,
  year filters, limit, and precision/recall mode.
- `read_full_document` reads OCR text for a specific Paperless document when
  the agent needs source detail beyond snippets.

The `/chat` UI uses a WebSocket endpoint so it can show turn state, tool-call
progress, final answers, and source cards. That makes the system easier to
debug and easier to trust: the user can see when the model searched, what kind
of source it used, and which documents back the answer.

> Media placeholder: add `assets/chat-demo.gif` or `assets/chat-demo.webm` here.
> Show a query that requires at least one tool call and a source-backed answer.

## Observability and Cost Management

Telemetry is exported through OpenTelemetry when `OTEL_EXPORTER_OTLP_ENDPOINT`
is set. The shared telemetry helper instruments LiteLLM and LangChain, and the
application adds spans around retrieval and tool execution. Phoenix then becomes
the shared place to inspect chat turns, model calls, token counts, tool latency,
retrieval sizes, and evaluation experiments.

> Media placeholder: add `assets/phoenix-trace.png` here.
> Show one redacted trace with chat, tool, retrieval, and LLM spans.

That observability closes the loop between product behavior and model cost. A
chat answer is not just a string; it is a traceable sequence of search, rerank,
read, and model calls. When a model configuration becomes too slow, too
expensive, or too inaccurate, the evaluation and tracing setup provide evidence
for changing that configuration instead of relying on anecdotes.
