# LinkedIn Post Series: docker-paperless-ai

This file contains drafts for a 3-part LinkedIn series about the paperless-ngx AI layer project, aimed at AI engineers, applied ML practitioners, and systems-minded builders.

**Posting schedule:** Monday, Wednesday, Friday (two weeks)
**Week 1:** Posts #1-3

---

## Post 1: The Setup

**Hook:** personal archive, keyword search limitation

**Draft:**

I have used paperless-ngx for years as my personal document archive.

It is excellent at the core document-management job: OCR, metadata extraction, tagging, and search for scanned PDFs and images, all on your own infrastructure.

What started as a few hundred files is now roughly 7,000 pages of paper mail and PDF attachments.

The problem is that search is still mostly keyword-based.

That works until you want to ask a normal human question:

"Show me all invoices from 2023 related to electricians."

Or:

"How much did I pay in federal taxes since 2022?"

Those are not keyword queries. They are retrieval + reasoning problems.

So I built an AI layer around paperless-ngx:

- batch OCR and metadata extraction
- semantic and hybrid retrieval
- an agentic chat system that can search, inspect documents, and answer with sources

The important design constraint was not "add AI everywhere."

It was:

- Paperless remains the system of record
- AI runs as an adjacent service
- retrieval stays grounded in source documents
- model choices are driven by evaluation, not hype

Over the next two posts, I’ll show the ingestion architecture and the agentic retrieval system behind it.

(Next: the ingestion pipeline that moves documents through OCR, metadata extraction, and embedding without coupling Paperless to model infrastructure.)

**Assets:** chat-demo.webm or chat-demo.png
**Hashtags:** #AIEngineering #RAG #OpenSource #SelfHosted #DocumentAI

---

## Post 2: The Architecture

**Hook:** ingestion architecture, not just model choice

**Draft:**

In the last post, I introduced the problem: paperless-ngx stores and OCRs documents well, but answering higher-level questions needs more than keyword search.

This post is about the ingestion architecture behind that AI layer.

Most AI demos focus on the model.

In practice, the harder problem is usually the system around it.

Here is the shape I ended up with:

- Paperless stays the system of record
- workflow tags define stage transitions: `ai:run-ocr`, `ai:run-metadata`, `ai:run-embed`
- a thin webhook listener converts Paperless events into Redis queue entries
- independent workers run OCR, metadata extraction, and embedding
- vectors go to Qdrant, traces go to Phoenix

Why this design works:

1. **Decoupled orchestration**
Paperless does not need to know anything about GPUs, vector databases, or model providers. It stores documents, emits events, and remains authoritative.

2. **Stage-level fault isolation**
OCR, metadata extraction, and embedding fail differently and have different compute profiles. Splitting them makes retries, debugging, and provider changes much easier.

3. **Evaluation-driven boundaries**
I did not choose the local/cloud split by intuition. I built an evaluation framework to compare OCR and metadata options by quality, cost, and operational fit for each task.

That led to a practical compromise:
- keep expensive, high-volume work local
- use hosted models where reliability and output quality matter more than a tiny per-document API bill

That was enough to backfill about 2,000 documents / 7,000 pages for under $1 in API cost.

4. **Operational realism**
My GPU workstation is not always on. The queue absorbs that constraint instead of assuming every model endpoint is permanently available.

To me, this is the real AI systems question:

**not just "which model is best?" but "what architecture keeps working when the environment is imperfect?"**

(Diagram below. Next: the agentic chat system and how it uses tools to choose its own retrieval path.)

**Assets:** data-ingestion-flow.png
**Hashtags:** #AIEngineering #MLOps #RAG #SystemDesign #OpenSource

---

## Post 3: The Agentic Chat System

**Hook:** agentic systems are interesting when they can choose their own path

**Draft:**

In the last post, I showed the ingestion pipeline.

This one is about the retrieval side: the agentic chat system.

This is the part I find most interesting, because it is where the system starts to feel genuinely agentic.

The chat layer is not a single prompt over a giant context window.

It is a tool-using agent that can decide how to gather evidence before it answers.

Its toolset is intentionally small:

- `get_available_metadata`
- `search_documents`
- `read_full_document`

The interesting part is not the number of tools.

It is that the agent can choose its own path.

A good answer might require:

- checking metadata first so filters match the real schema
- running hybrid retrieval over keywords and vectors
- reranking candidates locally
- deciding whether snippets are enough or whether full documents need to be read
- stopping only when the evidence is strong enough

That is the kind of agentic behavior I care about:

**not just tool calling, but adaptive path selection under uncertainty.**

The route is not fixed.

Sometimes metadata + retrieval is enough.
Sometimes the agent needs to inspect full source documents.
Sometimes the query needs precision.
Sometimes it needs broad recall.

So the agent can adapt its strategy:

- precision mode for exact facts or specific documents
- recall mode for broader surveys
- deeper reads only when snippets are insufficient

That matters because the agent is effectively choosing:

- how much evidence to gather
- how expensive retrieval should be
- how targeted or broad the search should become

This is why I prefer agentic retrieval over "dump everything into context and hope":

1. **Less wasted context**
The model fetches what it needs instead of receiving everything up front.

2. **Inspectable reasoning**
You can see which tools were called, in what order, and which documents supported the answer.

3. **Grounded answers**
Retrieval, reranking, and document reads stay deterministic. The LLM plans the path, but the evidence comes from tools.

To me, this is when agentic systems become genuinely useful:

when they can autonomously gather information, optimize their retrieval path, and converge on an answer with traceable evidence.

(Diagram below.)

**Assets:** agentic-chat-flow.png
**Hashtags:** #AgenticAI #AIEngineering #RAG #SystemDesign #OpenSource

---

## Quick Notes

**Engagement tips:**
- Respond to comments within 24 hours while thread is fresh
- Pin a comment with the repo link on post #3
- Ask a question in each post to encourage discussion

**Timing:**
- Best LinkedIn posting windows: 9-11am or 12-2pm (local time)
- Avoid weekends for posts #2-3

**Metrics to track:**
- Views, likes, comments, reposts per post
- Click-through rate on repo link
- New followers/contributors after thread completion
