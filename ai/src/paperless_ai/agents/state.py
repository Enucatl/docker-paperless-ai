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

    # Ordered list of 0-based page indices selected for vision OCR.
    # May be a subset of all pages for long documents (see ocr_first_pages /
    # ocr_last_pages / ocr_page_limit_threshold in AgentConfig).
    ocr_page_indices: list[int]

    # current_page is an index into ocr_page_indices, NOT a raw page number.
    # It advances by batch_size each iteration of the batched-vision-OCR loop.
    current_page: int
    batch_size: int

    # Accumulated across loop iterations — LangGraph concatenates via operator.add
    extracted_text_chunks: Annotated[list[str], operator.add]

    # Written by extract_metadata node; read back in SmartDocumentAgent.process()
    _extracted_metadata: dict
    _full_text: str
