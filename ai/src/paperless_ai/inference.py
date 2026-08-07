"""Application-facing access to the shared inference client."""

import os
from functools import lru_cache
from typing import Any

from shared_inference import CompletionResult, InferenceClient


def _endpoint(endpoint: str | None) -> str:
    return (
        endpoint
        or os.environ.get("INFERENCE_CHAT_ENDPOINT")
        or os.environ.get("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    ).rstrip("/")


def _api_key(endpoint: str) -> str | None:
    if "openrouter.ai" in endpoint:
        return os.environ.get("OPENROUTER_API_KEY")
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")


@lru_cache(maxsize=32)
def get_client(endpoint: str, domain: str) -> InferenceClient:
    """Return one shared client/session per endpoint and domain."""
    return InferenceClient(
        base_url=endpoint,
        api_key=_api_key(endpoint),
        provider="openrouter" if "openrouter.ai" in endpoint else "openai-compatible",
        domain=domain,
    )


async def complete(
    *,
    model: str,
    messages: list[dict[str, Any]],
    endpoint: str | None = None,
    domain: str,
    **kwargs: Any,
) -> CompletionResult:
    """Send one non-streaming completion, dropping legacy routing-only options."""
    kwargs.pop("num_retries", None)
    kwargs.pop("metadata", None)
    return await get_client(_endpoint(endpoint), domain).complete(
        model=model.removeprefix("openrouter/"), messages=messages, **kwargs
    )
