"""
Chunk situating for semantic embeddings.

``situate_chunks`` is the single public entry point.  It returns one situated
string per input chunk — context prepended to the raw text — using whichever
strategy is appropriate:

Resolution order
----------------
1. **Custom hook file** (``EMBED_HOOK_FILE`` env var) — if set and loadable,
   the file's ``situate_chunks`` coroutine is called and its result returned.
   The file is loaded once at process start and cached.

2. **Tier 2 — per-chunk LLM situating** (``config.situation_model`` set) —
   one LLM call per chunk, all fired concurrently via ``asyncio.gather``.
   The full document appears in a ``<document>`` block at the start of every
   prompt; providers with prefix caching (Anthropic ``cache_control``, vLLM
   automatic KV cache) reuse those tokens across the concurrent requests.

3. **Tier 1 — static metadata header** (default, zero extra LLM calls) —
   prepends Title / Sender / Date and, when available, the document Summary
   extracted during the metadata stage.  The summary is "free" because the
   metadata LLM already reads the whole document.

Custom hook file API
--------------------
The file must export a single coroutine::

    async def situate_chunks(
        chunks: list[str],
        full_text: str,
        meta,    # has .title, .correspondent, .document_date, .summary
        config,  # AgentConfig instance
    ) -> list[str]: ...
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from paperless_ai.agents.base import DocumentMetadata
    from paperless_ai.core.config import AgentConfig

_SITUATION_PROMPT = """\
<document>
{full_text}
</document>

Here is a chunk from this document:
<chunk>
{chunk}
</chunk>

Give a short context (1–2 sentences) situating this chunk within the overall \
document for the purposes of improving search retrieval. \
Answer only with the succinct context and nothing else.\
"""

# Module-level cache.
# _hook_resolved=True + _cached_hook=None  → checked, no custom hook file found.
# _hook_resolved=True + _cached_hook=<fn>  → custom hook loaded and cached.
_cached_hook: Optional[Callable] = None
_hook_resolved: bool = False


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _default_situate_chunks(
    chunks: list[str],
    full_text: str,
    meta: "DocumentMetadata",
    config: "AgentConfig",
) -> list[str]:
    """Tier 1: prepend a static metadata header to every chunk."""
    lines = [
        f"Title: {meta.title or 'Unknown'}",
        f"Sender: {meta.correspondent or 'Unknown'}",
        f"Date: {meta.document_date or 'Unknown'}",
    ]
    if getattr(meta, "summary", None):
        lines.append(f"Summary: {meta.summary}")
    lines.append("---")
    header = "\n".join(lines) + "\n"
    return [f"{header}{chunk}" for chunk in chunks]


async def _situate_single_chunk(
    chunk: str,
    ctx_text: str,
    config: "AgentConfig",
) -> str:
    """Tier 2: one LLM call that situates a single chunk within the document.

    The ``<document>`` block is identical for every chunk in the same document,
    so providers with prefix caching reuse those KV entries across the
    concurrent ``asyncio.gather`` calls in ``situate_chunks``.
    """
    import litellm
    from paperless_ai.core.telemetry import add_litellm_metadata

    prompt = _SITUATION_PROMPT.format(full_text=ctx_text, chunk=chunk)
    kwargs: dict = {
        "model": config.situation_model,
        "messages": [{"role": "user", "content": prompt}],
        **config.get_situation_litellm_kwargs(),
    }
    if config.situation_api_base:
        kwargs["api_base"] = config.situation_api_base
    add_litellm_metadata(
        kwargs,
        stage="embedding",
        operation="situate_chunk",
    )

    response = await litellm.acompletion(**kwargs)
    context = response.choices[0].message.content.strip()
    return f"{context}\n\n{chunk}"


def _resolve_hook() -> Optional[Callable]:
    """Load and cache the custom EMBED_HOOK_FILE hook; return None if absent.

    Returns None for "no custom hook" so callers can fall through to built-in
    tiers.  The _hook_resolved flag prevents repeated file-system hits.
    """
    import os

    global _cached_hook, _hook_resolved

    if _hook_resolved:
        return _cached_hook

    _hook_resolved = True

    hook_path = os.environ.get("EMBED_HOOK_FILE") or None
    if not hook_path:
        return None

    if not os.path.isfile(hook_path):
        log.warning(
            "EMBED_HOOK_FILE=%r does not exist — using built-in situating", hook_path
        )
        return None

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_paperless_ai_embed_hook", hook_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {hook_path!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        hook_fn = getattr(module, "situate_chunks", None)
        if hook_fn is None:
            raise AttributeError(f"{hook_path!r} does not define 'situate_chunks'")

        log.info("Loaded custom situate_chunks hook from %s", hook_path)
        _cached_hook = hook_fn
    except Exception:
        log.exception(
            "Failed to load embed hook from %r — using built-in situating", hook_path
        )

    return _cached_hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def situate_chunks(
    chunks: list[str],
    full_text: str,
    meta: "DocumentMetadata",
    config: "AgentConfig",
) -> list[str]:
    """Situate all chunks and return one situated string per chunk.

    See module docstring for resolution order and custom hook file API.
    Always returns a list of the same length as ``chunks``.
    """
    if not chunks:
        return []

    custom = _resolve_hook()
    if custom is not None:
        return await custom(chunks, full_text, meta, config)

    if config.situation_model:
        ctx = (
            full_text[: config.situation_context_chars]
            if config.situation_context_chars > 0
            else full_text
        )
        log.info("Situating %d chunk(s) via per-chunk LLM calls", len(chunks))
        return list(
            await asyncio.gather(
                *(_situate_single_chunk(c, ctx, config) for c in chunks)
            )
        )

    return await _default_situate_chunks(chunks, full_text, meta, config)
