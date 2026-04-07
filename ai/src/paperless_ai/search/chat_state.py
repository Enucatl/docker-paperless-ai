"""LangGraph state for the Paperless chat copilot."""

import operator
from typing import Annotated, TypedDict


class ChatState(TypedDict):
    """Conversation state passed between the agent and tool nodes."""

    messages: Annotated[list[dict], operator.add]
