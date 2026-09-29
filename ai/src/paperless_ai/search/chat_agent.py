"""Tool-calling Paperless chat copilot."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import niquests
from shared_inference import CompletionResult

from paperless_ai.core.config import AgentConfig
from paperless_ai.inference import complete
from paperless_common.paperless import PaperlessClient
from paperless_common.telemetry import set_span_attributes, start_span
from paperless_ai.search.tools import (
    TOOL_SCHEMAS,
    ToolExecutionResult,
    execute_tool_call_detailed,
    parse_tool_arguments,
    tool_validation_error,
)

SYSTEM_PROMPT = (
    "You are an AI assistant for a Paperless-ngx repository. "
    "Use tools to search documents and inspect source text before answering specific factual questions. "
    "Always cite relevant document IDs in your answer. "
    "Search uses Paperless full-text keywords: send short, distinctive terms rather than a natural-language sentence. "
    "Paperless query syntax supports uppercase AND, OR, and NOT, with parentheses for grouping; "
    "terms without an operator are implicitly combined with AND. Use OR for alternate words "
    "(for example, zoo OR tickets OR admission), and use AND only when all concepts must appear. "
    "Use double quotes for an exact phrase. "
    "Expand searches progressively; do not translate every term into every observed language. "
    "Start with distinctive names, identifiers, and keywords in the most plausible document language. "
    "Use the user's context first and document-language counts as a secondary guide; "
    "the language of the question need not be the language of the document. "
    "Use at most two languages per query and at most three alternatives per concept, joined with OR. "
    "If returned documents do not answer the question, even when there are matches, "
    "try simpler terms or another likely language in a separate short query. "
    "For exhaustive requests, cover the remaining relevant observed languages in separate "
    "short queries, combine results by document ID, and do not claim completeness if coverage is partial. "
    "Preserve proper names, brands, and identifiers. Language tags may cover only part of the archive: "
    "use them to guide query wording, and only filter by language tags when the user requests it. "
    "If observed languages are unknown, start with the user's keywords and adapt to the document text. "
    "If a keyword search returns nothing, retry with simpler keywords or another distinctive term before concluding there are no matches. "
    "Before filtering by metadata like tags, correspondents, document types, or storage paths, "
    "call get_available_metadata to confirm the exact names. "
    "Metadata filters require exact matches. If a filtered search returns no documents, retry "
    "without the restrictive metadata filter before concluding that no matching documents exist. "
    "Use search_documents with mode=precision for singular lookups and fact-finding, and "
    "mode=recall for exhaustive listing requests. When using mode=recall, always pass an explicit "
    "limit large enough for the requested count. After a precision search, read the top relevant "
    "document(s) before answering if the answer depends on document contents. Read more than one "
    "when multiple candidates remain plausible."
)

log = logging.getLogger(__name__)


def _snippet(text: str, limit: int = 320) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _extract_usage(response: CompletionResult) -> dict[str, int] | None:
    """Normalize optional token counts from the shared inference result."""
    usage = response.usage
    if all(
        count is None
        for count in (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
    ):
        return None
    return {
        "prompt_tokens": usage.prompt_tokens or 0,
        "completion_tokens": usage.completion_tokens or 0,
        "total_tokens": usage.total_tokens
        if usage.total_tokens is not None
        else (usage.prompt_tokens or 0) + (usage.completion_tokens or 0),
    }


@dataclass
class ChatTurnResult:
    reply: str
    history: list[dict]
    sources: dict[int, dict[str, bool]] = field(default_factory=dict)
    usage: dict[str, int] | None = None
    tool_activity: list[dict[str, Any]] = field(default_factory=list)


EventCallback = Callable[[dict[str, Any]], Awaitable[None]]


class ChatCopilot:
    """A small ReAct loop backed by the shared inference client."""

    def __init__(
        self,
        config: AgentConfig,
        client: PaperlessClient,
    ):
        self._config = config
        self._client = client

    async def _emit(
        self, callback: EventCallback | None, event: dict[str, Any]
    ) -> None:
        if callback is not None:
            await callback(event)

    def _model_kwargs(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        model = self._config.chat_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "tool_choice": "auto",
            **self._config.get_chat_kwargs(),
        }
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0.0
        endpoint = self._config.chat_endpoint
        if endpoint:
            kwargs["endpoint"] = endpoint
        return kwargs

    @staticmethod
    def _merge_source_flags(
        sources: dict[int, dict[str, bool]], result: ToolExecutionResult
    ) -> None:
        for ref in result.source_refs:
            state = sources.setdefault(
                ref.doc_id, {"matched": False, "inspected": False}
            )
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
        client = self._client.metadata_snapshot()
        with start_span(
            "paperless_ai.chat.turn",
            **{
                "paperless_ai.chat.model": self._config.chat_model,
                "paperless_ai.chat.history_messages": len(history or []),
                "paperless_ai.chat.user_message_length": len(user_message),
            },
        ) as turn_span:
            try:
                languages = await client.get_document_languages()
            except niquests.RequestException:
                log.warning("Could not load document languages for chat", exc_info=True)
                languages = {}
            language_context = (
                ", ".join(
                    f"{code}: {count} documents" for code, count in languages.items()
                )
                or "unknown (no language inventory available)"
            )
            messages = [
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\nObserved document languages (ISO codes and document counts): {language_context}.",
                }
            ]
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": user_message})
            aggregated_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            sources: dict[int, dict[str, bool]] = {}
            tool_activity: list[dict[str, Any]] = []

            while True:
                await self._emit(
                    event_callback,
                    {
                        "type": "status",
                        "phase": "model",
                        "content": "Thinking about the next step.",
                    },
                )
                response = await complete(
                    domain="browser_copilot", **self._model_kwargs(messages)
                )
                usage = _extract_usage(response)
                model = self._config.chat_model
                if usage is None:
                    await self._emit(
                        event_callback,
                        {
                            "type": "usage",
                            "scope": "step",
                            "model": model,
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
                            "model": model,
                            "available": True,
                            **usage,
                        },
                    )

                assistant_message = response.message
                messages.append(assistant_message)
                tool_calls = assistant_message.get("tool_calls") or []
                set_span_attributes(
                    turn_span,
                    **{
                        "paperless_ai.chat.tool_call_count": len(tool_calls),
                        "paperless_ai.chat.prompt_tokens": aggregated_usage[
                            "prompt_tokens"
                        ],
                        "paperless_ai.chat.completion_tokens": aggregated_usage[
                            "completion_tokens"
                        ],
                        "paperless_ai.chat.total_tokens": aggregated_usage[
                            "total_tokens"
                        ],
                    },
                )
                if not tool_calls:
                    assistant_text = str(assistant_message.get("content") or "").strip()
                    total_usage = (
                        aggregated_usage if aggregated_usage["total_tokens"] else None
                    )
                    set_span_attributes(
                        turn_span,
                        **{
                            "paperless_ai.chat.final_sources": len(sources),
                            "paperless_ai.chat.reply_length": len(assistant_text),
                        },
                    )
                    return ChatTurnResult(
                        reply=assistant_text,
                        history=messages[1:],
                        sources=sources,
                        usage=total_usage,
                        tool_activity=tool_activity,
                    )

                for tool_call in tool_calls:
                    function = tool_call.get("function", {})
                    name = str(function.get("name") or "unknown_tool")
                    argument_error = None
                    try:
                        args = parse_tool_arguments(function.get("arguments"))
                    except ValueError as exc:
                        args = function.get("arguments")
                        argument_error = tool_validation_error(str(exc))
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
                    with start_span(
                        "paperless_ai.chat.tool_call",
                        **{
                            "paperless_ai.tool.name": name,
                            "paperless_ai.tool.call_id": str(tool_call.get("id") or ""),
                        },
                    ) as tool_span:
                        result = argument_error or await execute_tool_call_detailed(
                            name,
                            args,
                            client=client,
                        )
                        duration_ms = int((time.perf_counter() - start) * 1000)
                        set_span_attributes(
                            tool_span,
                            **{
                                "paperless_ai.tool.duration_ms": duration_ms,
                                "paperless_ai.tool.summary": result.summary,
                                "paperless_ai.tool.source_ref_count": len(
                                    result.source_refs
                                ),
                                "paperless_ai.tool.preview_length": len(
                                    result.preview or ""
                                ),
                            },
                        )
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
                    tool_activity.append(
                        {
                            "tool_call_id": tool_call.get("id"),
                            "name": name,
                            "arguments": args,
                            "summary": result.summary,
                            "preview": result.preview or _snippet(result.content),
                            "duration_ms": duration_ms,
                        }
                    )
                    await self._emit(
                        event_callback,
                        {
                            "type": "status",
                            "phase": "tool",
                            "content": result.summary,
                        },
                    )
