"""State for complete-document parsing and metadata extraction."""

from typing import TypedDict


class AgentState(TypedDict):
    """Document input, parsing output, and extracted metadata."""

    file_path: str
    total_pages: int
    extracted_text_chunks: list[str]
    _extracted_metadata: dict
    _full_text: str
    _metadata_context: str
