# Case Study: AI Document Copilot for paperless-ngx

> Media placeholder: add `assets/chat-demo.gif` or `assets/chat-demo.webm` here.
> The clip should show a redacted real query in the `/chat` UI, the tool-call
> panel, the final answer, and source cards.

This project turns a paperless-ngx archive into an AI-searchable document
system without patching paperless-ngx itself. New documents are imported through
the normal Paperless flow, then an external AI service re-OCRs the pages,
extracts structured metadata, indexes the content for semantic retrieval, and
serves a browser copilot that can search, inspect, and answer questions over the
archive.

The goal was not just to attach an LLM to a document store. The useful product
boundary was a dependable local deployment that can process private documents,
recover when a model server is unavailable, support both cloud and self-hosted
models, and make model quality visible through evaluation and tracing.

## What It Does

- Re-OCRs imported documents with a vision model and writes the transcript back
  to Paperless.
- Extracts title, date, correspondent, summary, and structured debug output via
  a metadata model.
- Indexes document chunks in Qdrant with dense and sparse bge-m3 vectors.
- Provides a `/search` endpoint for ranked document IDs.
- Provides a browser chat copilot that can call tools, search the archive, read
  source text, and return source-backed answers.
- Supports cloud models through LiteLLM and local OpenAI-compatible endpoints
  such as vLLM.
- Exports tracing to Phoenix and includes a Phoenix-backed evaluation workflow
  for OCR and metadata extraction experiments.

## Architecture

> Media placeholder: add `assets/architecture-overview.png` here.
> The diagram should show Paperless workflows, the webhook listener, Redis,
> AI workers, model endpoints, Qdrant, the chat/search service, and Phoenix.

The system is deliberately built around the Paperless API and workflow model.
Paperless remains the source of truth for documents and metadata. The AI layer
is an adjacent service that reacts to workflow tags, moves documents through
independent stages, and writes results back through supported REST APIs.

The ingestion path is:

1. Paperless imports a document and assigns `ai:run-ocr`.
2. The webhook listener receives the Paperless event and enqueues the document
   ID in Redis.
3. The AI service downloads the original PDF, sends page images to the OCR
   model, and writes the transcript to the Paperless content field.
4. The metadata stage extracts document fields from the transcript and patches
   Paperless metadata.
5. The embedding stage chunks the document, writes vectors to Qdrant, and
   removes the stage tag.

This design accepts eventual consistency in exchange for operational isolation:
if a local GPU workstation or model endpoint is down, the worker can defer work
without corrupting Paperless state or blocking normal document ingestion.

## Why Evaluation Was Central

The project reached maturity because model choice was treated as an empirical
question. OCR and metadata extraction are easy to demo on a single clean
document, but a personal archive contains scans, forms, letters, dates in many
formats, missing correspondents, and documents where a plausible-looking answer
can still be wrong.

The repo includes a golden dataset workflow and Phoenix-backed experiment
runners. Experiments can compare model configurations for OCR and metadata
extraction, track metrics such as correspondent exact/fuzzy match and date
exact/partial match, and keep title quality evaluation separate from the model
being tested.

> Media placeholder: add `assets/eval-comparison.png` here.
> Use a redacted Phoenix comparison or terminal summary that shows multiple
> model configurations compared on the same dataset.

That evaluation loop made the model tradeoff explicit: local models are
attractive for privacy and predictable cost, but they must be judged against the
quality and latency of hosted models. The current architecture keeps model
endpoints configurable by stage, so OCR, metadata extraction, chat, and
embedding can be optimized independently instead of forcing one model to do
everything.

## Production Engineering Signals

The implementation is structured as a deployable service, not a notebook demo.
Important production-oriented decisions include:

- Separate listener and AI service boundaries: webhook ingress remains thin,
  while long-running inference and chat live in the AI service.
- Redis-backed queues and stage tags: documents can wait safely and retry
  without relying on a single in-process job.
- Failure handling: repeated failures move to a failed queue instead of
  retrying forever.
- Local search process lifecycle: query embedding and reranking models are
  lazy-loaded in a child process and released after an idle timeout to reclaim
  memory.
- API compatibility tests: the test suite documents niquests behavior and
  guards against accidentally introducing httpx-only parameters.
- Docker E2E tests: integration tests run against real Paperless, Redis, and
  Qdrant services.
- Phoenix telemetry: traces cover LLM calls, LangChain/LangGraph execution,
  retrieval, and tool calls where instrumentation is available.

> Media placeholder: add `assets/phoenix-trace.png` here.
> Show a redacted Phoenix trace for one chat turn or processing run, including
> tool spans and LLM token/cost fields where visible.

## Portfolio Takeaway

This project is useful as a portfolio piece because it connects applied AI with
the constraints that matter in production: data privacy, model evaluation,
fallback behavior, cost visibility, observability, and integration with an
existing product rather than a greenfield demo. The system demonstrates how to
turn an LLM prototype into a maintainable workflow around real documents and
real operational failure modes.
