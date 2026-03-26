"""
LangGraph state definition for the SmartDocumentAgent.

Uses TypedDict so the graph can merge partial updates from each node.
The `extracted_text_chunks` list uses operator.add annotation so LangGraph
automatically concatenates chunks from successive batched-vision-OCR loops.
"""

import operator
from typing import Annotated, Optional, TypedDict


class AgentState(TypedDict):
    # Input
    file_path: str
    language: Optional[str]

    # Set by analyze_pdf node
    total_pages: int
    is_digital_text: bool

    # Advances each iteration of the batched-vision-OCR loop
    current_page: int
    batch_size: int

    # Accumulated across loop iterations — LangGraph concatenates via operator.add
    extracted_text_chunks: Annotated[list[str], operator.add]
