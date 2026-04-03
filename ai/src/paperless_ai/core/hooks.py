"""
Dynamic embed hook loader.

By default, each chunk is prefixed with a structured context header before
being sent to the embedding model (situated embeddings). Advanced users can
replace this behaviour by mounting a custom Python file and pointing
EMBED_HOOK_FILE at it.

The custom file must expose a single coroutine with this exact signature:

    async def format_chunk_for_embedding(chunk: str, meta, config) -> str: ...

where ``meta`` has ``.title``, ``.correspondent``, and ``.document_date``
attributes, and ``config`` is the live ``AgentConfig`` instance.

The loaded function is cached at the module level so the file is parsed only
once at process startup.
"""

import logging
import os
from typing import Awaitable, Callable, Optional

log = logging.getLogger(__name__)

# TYPE_CHECKING imports only — avoids a circular dependency at runtime.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paperless_ai.agents.base import DocumentMetadata
    from paperless_ai.core.config import AgentConfig

# Module-level cache: None means "not yet resolved".
_cached_hook: Optional[Callable] = None
_hook_resolved = False


async def default_embed_hook(chunk: str, meta: "DocumentMetadata", config: "AgentConfig") -> str:
    """Prepend a structured context header to the chunk (default situated embedding)."""
    header = (
        f"Title: {meta.title or 'Unknown'}\n"
        f"Sender: {meta.correspondent or 'Unknown'}\n"
        f"Date: {meta.document_date or 'Unknown'}\n"
        f"---\n"
    )
    return f"{header}{chunk}"


def get_embed_hook() -> Callable[[str, "DocumentMetadata", "AgentConfig"], Awaitable[str]]:
    """Return the active embed hook coroutine, loading it once from EMBED_HOOK_FILE if set.

    Resolution order:
    1. Return cached result on all calls after the first.
    2. If EMBED_HOOK_FILE is unset or the file is missing, use default_embed_hook.
    3. Dynamically import the file, extract format_chunk_for_embedding, cache and return it.
    4. On any import error, log the exception and fall back to default_embed_hook.
    """
    global _cached_hook, _hook_resolved

    if _hook_resolved:
        return _cached_hook  # type: ignore[return-value]

    _hook_resolved = True

    hook_path = os.environ.get("EMBED_HOOK_FILE")
    if not hook_path:
        _cached_hook = default_embed_hook
        return _cached_hook

    if not os.path.isfile(hook_path):
        log.warning(
            "EMBED_HOOK_FILE is set to %r but the file does not exist — using default hook",
            hook_path,
        )
        _cached_hook = default_embed_hook
        return _cached_hook

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_paperless_ai_embed_hook", hook_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec for {hook_path!r}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        hook_fn = getattr(module, "format_chunk_for_embedding", None)
        if hook_fn is None:
            raise AttributeError(
                f"{hook_path!r} does not define 'format_chunk_for_embedding'"
            )

        log.info("Loaded custom embed hook from %s", hook_path)
        _cached_hook = hook_fn
    except Exception:
        log.exception(
            "Failed to load embed hook from %r — falling back to default", hook_path
        )
        _cached_hook = default_embed_hook

    return _cached_hook
