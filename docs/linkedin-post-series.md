# LinkedIn Post Series: docker-paperless-ai

This file contains drafts for a 6-part LinkedIn thread about the paperless-ngx AI layer project.

**Posting schedule:** Monday, Wednesday, Friday (two weeks)
**Week 1:** Posts #1-3 | **Week 2:** Posts #4-6

---

## Post 1: The Setup

**Hook:** personal archive, keyword search limitation

**Draft:**

For years, I've been using paperless-ngx as my personal document archive.

It's a self-hosted document management system where you drop in scanned PDFs and images, and it OCRs everything, extracts metadata, and makes it searchable. Think: Google Drive for your physical mail, but on your own server.

Started with a few hundred documents. Now sitting on ~7,000 pages of personal docs from years of paper mail and email PDFs.

Here's the problem: despite all that OCR, **search is still keyword-based**. Ask it something like "show me all invoices from 2023 about electricians" and you're out of luck.

So I built a layer on top.

**AI batch OCR + semantic search + a chat copilot that can answer questions over your entire archive.**

In this thread: the architecture, the model stack, the tradeoffs, and why I chose stability over cleverness.

(Next: how it works without keeping 1-2GB models in RAM 24/7)

**Assets:** chat-demo.webm or chat-demo.png
**Hashtags:** #AI #OpenSource #RAG #SelfHosted #DocumentManagement

---

## Post 2: The Architecture

**Hook:** RAM-efficient lazy loading

**Draft:**

One of the trickiest parts: how do you serve semantic search with a local reranker without keeping 1-2GB models in RAM 24/7?

The answer: **process-backed workers with lazy loading.**

When a query arrives:
- FastAPI checks if the local embedder/reranker worker is running
- If not, spawns a child process that loads the ~1GB model (~15s warmup)
- Subsequent queries reuse that process (~50ms latency)
- Idle for 5 minutes? The worker exits, reclaiming the RAM

This is critical because the GPU workstation that hosts the vision LLM isn't always running. Can't depend on it for the reranker, but can't afford to keep the model resident all the time either.

The result: good search quality on CPU when needed, minimal memory footprint when not.

(Next: the model stack and why each one matters)

**Assets:** data-ingestion-flow.png
**Hashtags:** #LLM #SystemDesign #Performance #EdgeAI #OpenSource

---

## Post 3: The Model Stack

**Hook:** hybrid approach, cost/quality balance

**Draft:**

Not all models are created equal. Here's the stack I landed on:

**OCR:** Nanonets-OCR2-3B running locally
- 10x cheaper than cloud OCR in token terms
- 40% faster in my setup
- Keeps page images from leaving the infrastructure

**Metadata extraction:** Gemini 3.1 flash-lite
- Complex text understanding that local models struggle with
- Still relatively cheap for the token volume we're talking about

**Embeddings:** bge-m3
- Dense + sparse vectors in one model
- Works with Qdrant's hybrid search for best results

**Reranking:** BAAI/bge-reranker-v2-m3 (~1GB)
- Local, CPU-based
- Crucial for query relevance, especially on mixed "precision vs recall" queries

Total API cost for a backfill of ~2,000 documents (7,000 pages): **under $1**

Because: page-image OCR and embeddings never hit the hosted model. Only metadata extraction does, and that's lightweight text.

(Next: the path not taken)

**Assets:** eval-comparison.png (Phoenix experiment chart)
**Hashtags:** #LLM #MachineLearning #CostOptimization #OpenSource

---

## Post 4: The Rejected Path

**Hook:** why cleverness wasn't worth it

**Draft:**

Qwen 3.5 9B fit my RTX 5090 hardware and showed promise. But it had a stubborn problem:

**When prompted to reason, the model would keep "thinking"** — emitting text between `<thinking>` and `</thinking>` tags — **until it exhausted its token budget** instead of closing the section naturally.

The technical fix was straightforward but intrusive: inject custom logic into the logit generation loop to gradually bias the model toward emitting `</thinking>` once it hit a predefined token limit. This requires:
- Intercepting the generation stream
- Modifying probability distributions on-the-fly
- Managing state across generation steps

I could have implemented that fix. But it would mean carrying custom generation code in this project — a maintenance burden for a personal pipeline.

The chosen tradeoff: **use a stable model stack (Gemini flash-lite) that works reliably out of the box**, even if it costs a bit more. For a 1-2 documents/week workload, the extra API cost is negligible compared to the time saved not maintaining model-specific hacks.

Sometimes the boring choice is the right engineering decision.

(Next: the results)

**Assets:** full-metadata-trace.png (Phoenix trace)
**Hashtags:** #EngineeringTradeoffs #LLM #SystemDesign #OpenSource

---

## Post 5: The Result

**Hook:** concrete numbers, ROI

**Draft:**

2,000 documents processed
7,000 pages OCR'd
**Less than $1 in API costs**

The numbers come from a backfill of my entire archive, using the final model stack (Nanonets OCR + Gemini metadata + bge-m3 embeddings).

But the real ROI isn't just money. It's **time**:
- 15 minutes to ask a question about any document in the archive
- Source-backed answers with links to the original files
- No manual tagging or metadata entry

The privacy-conscious option: run everything on-prem with Ollama. The cost is setup complexity and GPU requirements. For most people, the hybrid approach (local OCR + cloud metadata) is the sweet spot.

And yes, if you're wondering: the chat model can't hallucinate sources. Every answer includes citations with clickable links back to the Paperless document.

(Next: open source)

**Assets:** phoenix_trace.png or screenshot of search query + results
**Hashtags:** #DIY #Privacy #ROI #OpenSource #RAG

---

## Post 6: Open Source

**Hook:** no patches, easy to try

**Draft:**

**No paperless-ngx patches required.**

The AI layer runs as an external service that:
- Reacts to Paperless workflow tags
- Moves documents through independent stages
- Writes results back through supported REST APIs

Paperless remains the source of truth. The AI is an adjacent service that enhances it.

If you're using paperless-ngx and want AI-powered search:
- **Try it:** `docker compose up ai` (see repo for setup)
- **Learn from it:** the architecture is documented in `docs/deep-dive.md`
- **Contribute:** open issues, PRs welcome

The code is at [github.com/Enucatl/docker-paperless-ai](https://github.com/Enucatl/docker-paperless-ai)

Thanks for reading this thread. If you find it useful, give it a repost. If you've built something similar, I'd love to hear about your approach in the comments.

**Assets:** repo screenshot or architecture overview
**Hashtags:** #OpenSource #PaperlessNGX #AI #SelfHosted #DeveloperCommunity

---

## Quick Notes

**Engagement tips:**
- Respond to comments within 24 hours while thread is fresh
- Pin a comment with the repo link on posts #4-6
- Ask a question in each post to encourage discussion

**Timing:**
- Best LinkedIn posting windows: 9-11am or 12-2pm (local time)
- Avoid weekends for posts #2-5 (keep weekend for #6, broader audience)

**Metrics to track:**
- Views, likes, comments, reposts per post
- Click-through rate on repo link
- New followers/contributors after thread completion

