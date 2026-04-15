# Architecture RAM Findings

Date of measurement: 2026-04-08 (UTC)

This note captures the current RAM profile of the Paperless AI stack and the
main architectural reasons for the observed memory usage.

## Current container memory usage

Measured with `docker stats --no-stream`:

- `paperless-ai-ai-1`: `361.5 MiB`
- `paperless-ai-webhook-listener-1`: `394.8 MiB`
- `paperless-ai-webserver-1`: `560.3 MiB`
- `paperless-ai-qdrant-1`: `44.9 MiB`
- `paperless-ai-broker-1`: `8.5 MiB`

At the time of inspection, the AI worker was not the largest consumer. The
webhook container was using nearly as much RAM as the worker.

## Main finding

The service named `webhook-listener` is not acting as a thin webhook ingress.
It is also the always-on chat and search service.

At module import and startup, [`webhook.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/webhook.py)
pulls in:

- `ChatCopilot`
- retrieval and search tools
- `litellm`
- `qdrant_client`
- `PaperlessClient`
- the local-search process manager

Relevant imports:

- [`webhook.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/webhook.py#L42)
- [`chat_agent.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/chat_agent.py#L9)
- [`chat_agent.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/chat_agent.py#L16)
- [`tools.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/tools.py#L9)
- [`tools.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/tools.py#L10)

This means the lightweight webhook path permanently pays for the chat/search
dependency graph.

## Import-time memory baseline

The webhook memory cost is largely baseline import cost, not an obvious leak.

Measured with `python -c` inside the running containers:

- `import litellm`: about `151560 KB` RSS
- `import qdrant_client`: about `96624 KB` RSS
- `import paperless_ai.search.chat_agent`: about `211296 KB` RSS
- `import paperless_ai.search.webhook`: about `217976 KB` RSS

Smaller reference points:

- `from paperless_ai.search.queue import TaskQueues`: about `33464 KB`
- `from paperless_ai.core.paperless import PaperlessClient`: about `36368 KB`
- `import fastapi, uvicorn`: about `44752 KB`

Conclusion: the webhook container is heavy mainly because the chat/search stack
is imported into the same always-on process.

## AI worker memory profile

The AI worker is also carrying a large import baseline because it uses the OCR
and metadata pipeline built around `litellm` and the smart agent stack.

Measured imports:

- `import litellm`: about `151560 KB` RSS
- `from paperless_ai.search.embedder import EmbeddingAPIEmbedder`: about `161056 KB`
- `from paperless_ai.core import runner`: about `161940 KB`
- `import paperless_ai.agents.smart_graph_agent`: about `180188 KB`

The observed steady-state worker usage of about `361 MiB` is therefore
consistent with the current dependency set and long-lived Python process model.

## Chat-triggered local model loading

There are two distinct webhook memory regimes:

1. Baseline webhook app memory from imports and startup
2. Additional temporary memory when the chat UI warms local search models

Webhook logs from 2026-04-08 show:

- `GET /chat`
- WebSocket connection to `/ws/chat`
- local-search warmup triggered by `LOCAL_SEARCH_WARM_ON_CHAT_LOAD=true`
- child process loading `BAAI/bge-m3`
- child process loading `BAAI/bge-reranker-v2-m3`
- worker exit after the configured idle timeout

Relevant runtime settings:

- [`docker-compose.yml`](/opt/docker/paperless-ai/docker-compose.yml)
- [`webhook.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/webhook.py#L118)

The logs also showed:

- `local_search_idle_timeout=180s`
- `warm_on_startup=False`
- `warm_on_chat_load=True`

So a single visit to the chat page can temporarily add the local embedding and
reranking models in a child process, even though that extra memory later scales
back down after the idle timeout.

## Traffic observations

The webhook container had much higher cumulative network and block I/O than the
AI worker during inspection:

- network receive: about `4.6 GB`
- block write: about `6.98 GB`

This supports the conclusion that the service is handling real interactive UI
traffic, not just occasional Paperless webhook events.

## Architectural issue

The current design combines two different concerns in one container:

- thin document-event ingress
- interactive copilot and search application

Those concerns have very different resource profiles. The ingress path should
be cheap and always on. The chat/search path is inherently heavier because it
brings in `litellm`, search tools, Qdrant access, and optional local model
warmup.

## Recommended next changes

### Best structural fix

Split the current `webhook-listener` into two services:

- `webhook-ingress`
  - only `/webhook/document`
  - only `/health`
  - only Redis queue routing
- `paperless-copilot`
  - `/chat`
  - `/ws/chat`
  - `/search`
  - metadata browsing

This would let the webhook ingress stay lightweight while the chat/search
service can be started, scaled, or constrained independently.

### Lower-risk incremental fix

If a full split is not desirable immediately, lazy-load the heavy chat/search
stack:

- move chat/search imports out of module top-level in
  [`webhook.py`](/opt/docker/paperless-ai/ai/src/paperless_ai/search/webhook.py#L42)
- only initialize `ChatCopilot`, retrieval, and Qdrant access on first use of
  `/chat`, `/ws/chat`, or `/search`

This would not make chat cheap, but it would avoid charging the webhook-only
path for those imports.

### Optional tuning

If lower idle RAM matters more than first-query latency, disable automatic chat
warmup:

- set `LOCAL_SEARCH_WARM_ON_CHAT_LOAD=false`

This will not remove the import-time memory cost, but it will reduce temporary
RAM spikes caused by eager loading of local search models when the chat page is
opened.

## Bottom line

The current RAM usage does not primarily look like a memory leak. It looks like
an architectural coupling issue:

- `ai` is a heavy long-lived worker by design
- `webhook-listener` is heavy because it is actually a copilot/search service
  plus webhook ingress in one process

If the goal is to reduce overall steady-state RAM, the highest-value change is
to separate thin webhook ingress from interactive search/chat.
