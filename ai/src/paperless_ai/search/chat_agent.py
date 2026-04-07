"""LangGraph-based Paperless chat copilot."""

from __future__ import annotations

from typing import Any

import litellm

from paperless_ai.core.config import AgentConfig
from paperless_ai.core.paperless import PaperlessClient
from paperless_ai.search.chat_state import ChatState
from paperless_ai.search.embedder import LocalLazySearchEmbedder
from paperless_ai.search.tools import TOOL_SCHEMAS, execute_tool_call, parse_tool_arguments

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


class ChatCopilot:
    """A small ReAct loop backed by LangGraph and LiteLLM."""

    def __init__(
        self,
        config: AgentConfig,
        client: PaperlessClient,
        embedder: LocalLazySearchEmbedder,
        qdrant_url: str,
    ):
        self._config = config
        self._client = client
        self._embedder = embedder
        self._qdrant_url = qdrant_url
        self._graph = self._build_graph()

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        async def agent_node(state: ChatState) -> dict:
            kwargs: dict[str, Any] = {
                "model": self._config.effective_metadata_model,
                "messages": state["messages"],
                "tools": TOOL_SCHEMAS,
                "tool_choice": "auto",
                **self._config.get_metadata_litellm_kwargs(),
            }
            kwargs["temperature"] = 0.0
            if self._config.metadata_api_base:
                kwargs["api_base"] = self._config.metadata_api_base
            response = await litellm.acompletion(**kwargs)
            return {"messages": [_message_to_dict(response.choices[0].message)]}

        async def tool_node(state: ChatState) -> dict:
            last_message = state["messages"][-1]
            tool_results = []
            for tool_call in last_message.get("tool_calls", []):
                function = tool_call.get("function", {})
                name = function.get("name")
                args = parse_tool_arguments(function.get("arguments"))
                result = await execute_tool_call(
                    str(name),
                    args,
                    client=self._client,
                    embedder=self._embedder,
                    qdrant_url=self._qdrant_url,
                )
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": name,
                        "content": result,
                    }
                )
            return {"messages": tool_results}

        workflow = StateGraph(ChatState)
        workflow.add_node("agent_node", agent_node)
        workflow.add_node("tool_node", tool_node)
        workflow.set_entry_point("agent_node")
        workflow.add_conditional_edges(
            "agent_node",
            route_tools,
            {"tool_node": "tool_node", "__end__": END},
        )
        workflow.add_edge("tool_node", "agent_node")
        return workflow.compile()

    async def run_turn(self, user_message: str, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Run one user turn and return the assistant reply plus updated history."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        final_state = await self._graph.ainvoke({"messages": messages})
        final_messages = final_state["messages"]
        assistant_message = final_messages[-1]
        assistant_text = str(assistant_message.get("content") or "").strip()
        return assistant_text, final_messages[1:]
