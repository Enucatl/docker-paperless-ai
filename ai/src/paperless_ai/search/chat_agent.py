"""LangGraph-based Paperless chat copilot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import litellm

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.search.chat_state import ChatState
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.tools import TOOL_SCHEMAS, ToolExecutionResult, execute_tool_call_detailed, parse_tool_arguments

SYSTEM_PROMPT = (
    "You are an AI assistant for a Paperless-ngx repository. "
    "Use tools to search documents and inspect source text before answering specific factual questions. "
    "Always cite relevant document IDs in your answer. "
    "Before filtering by metadata like tags, correspondents, document types, or storage paths, "
    "call get_available_metadata to confirm the exact names."
)


def route_tools(state: ChatState) -> str:
    """Route to the tool node when the assistant emitted tool calls."""
    last_message = state["messages"][-1]
    return "tool_node" if last_message.get("tool_calls") else "__end__"


def _message_to_dict(message: Any) -> dict:
    if isinstance(message, dict):
        return {k: v for k, v in message.items() if v is not None}
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported message type: {type(message).__name__}")


def _snippet(text: str, limit: int = 320) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if not usage:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    elif not isinstance(usage, dict):
        usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    normalized = {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or (prompt_tokens or 0) + (completion_tokens or 0)),
    }
    return normalized


@dataclass
class ChatTurnResult:
    reply: str
    history: list[dict]
    sources: dict[int, dict[str, bool]] = field(default_factory=dict)
    usage: dict[str, int] | None = None


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class ChatCopilot:
    """A small ReAct loop backed by LangGraph and LiteLLM."""

    def __init__(
        self,
        config: AgentConfig,
        client: PaperlessClient,
        embedder: LocalLazySearchEmbedder,
        qdrant_url: str,
        rerank_model: str | None = None,
        rerank_api_base: str | None = None,
    ):
        self._config = config
        self._client = client
        self._embedder = embedder
        self._qdrant_url = qdrant_url
        self._rerank_model = rerank_model
        self._rerank_api_base = rerank_api_base
    async def _emit(self, callback: EventCallback | None, event: dict[str, Any]) -> None:
        if callback is not None:
            await callback(event)

    def _model_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._config.effective_metadata_model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            **self._config.get_metadata_litellm_kwargs(),
        }
        kwargs["temperature"] = 0.0
        if self._config.metadata_api_base:
            kwargs["api_base"] = self._config.metadata_api_base
        return kwargs

    @staticmethod
    def _merge_source_flags(sources: dict[int, dict[str, bool]], result: ToolExecutionResult) -> None:
        for ref in result.source_refs:
            state = sources.setdefault(ref.doc_id, {"matched": False, "inspected": False})
            if ref.source_type == "search":
                state["matched"] = True
            if ref.source_type == "read":
                state["inspected"] = True

    async def run_turn(
        self,
        user_message: str,
        history: list[dict] | None = None,
        event_callback: EventCallback | None = None,
    ) -> ChatTurnResult:
        """Run one user turn and return the assistant reply, history, sources, and usage."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        aggregated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        sources: dict[int, dict[str, bool]] = {}

        while True:
            await self._emit(
                event_callback,
                {"type": "status", "phase": "model", "content": "Thinking about the next step."},
            )
            response = await litellm.acompletion(**self._model_kwargs(messages))
            usage = _extract_usage(response)
            if usage is None:
                await self._emit(
                    event_callback,
                    {
                        "type": "usage",
                        "scope": "step",
                        "model": self._config.effective_metadata_model,
                        "available": False,
                    },
                )
            else:
                for key, value in usage.items():
                    aggregated_usage[key] += value
                await self._emit(
                    event_callback,
                    {
                        "type": "usage",
                        "scope": "step",
                        "model": self._config.effective_metadata_model,
                        "available": True,
                        **usage,
                    },
                )

            assistant_message = _message_to_dict(response.choices[0].message)
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                assistant_text = str(assistant_message.get("content") or "").strip()
                total_usage = aggregated_usage if aggregated_usage["total_tokens"] else None
                return ChatTurnResult(
                    reply=assistant_text,
                    history=messages[1:],
                    sources=sources,
                    usage=total_usage,
                )

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = str(function.get("name") or "unknown_tool")
                args = parse_tool_arguments(function.get("arguments"))
                await self._emit(
                    event_callback,
                    {
                        "type": "tool_call_started",
                        "tool_call_id": tool_call.get("id"),
                        "name": name,
                        "arguments": args,
                    },
                )
                start = time.perf_counter()
                result = await execute_tool_call_detailed(
                    name,
                    args,
                    client=self._client,
                    embedder=self._embedder,
                    qdrant_url=self._qdrant_url,
                    rerank_model=self._rerank_model,
                    rerank_api_base=self._rerank_api_base,
                )
                duration_ms = int((time.perf_counter() - start) * 1000)
                self._merge_source_flags(sources, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result.content,
                    }
                )
                await self._emit(
                    event_callback,
                    {
                        "type": "tool_call_completed",
                        "tool_call_id": tool_call.get("id"),
                        "name": name,
                        "arguments": args,
                        "summary": result.summary,
                        "preview": result.preview or _snippet(result.content),
                        "duration_ms": duration_ms,
                    },
                )
                await self._emit(
                    event_callback,
                    {
                        "type": "status",
                        "phase": "tool",
                        "content": result.summary,
                    },
                )
