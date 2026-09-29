"""Document metadata and processing results."""

from typing import Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from a document."""

    title: Optional[str] = None
    document_date: Optional[str] = None  # ISO 8601: YYYY-MM-DD
    correspondent: Optional[str] = None
    summary: Optional[str] = None
    languages: list[str] | None = None
    full_ocr_transcript: str = ""


class AgentResult(BaseModel):
    """Result returned by any document agent."""

    metadata: DocumentMetadata
    metadata_context: str = ""
    elapsed_s: float = 0.0
    pages: int = 0
    chars: int = 0
    ocr_method: str = "layout-parsing"
