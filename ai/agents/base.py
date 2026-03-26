"""
Base types and abstract class for document processing agents.

All agents accept a file path on disk (memory-safe) and return an AgentResult
containing extracted DocumentMetadata and basic telemetry.
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Structured metadata extracted from a document."""

    title: Optional[str] = None
    document_date: Optional[str] = None  # ISO 8601: YYYY-MM-DD
    correspondent: Optional[str] = None
    full_ocr_transcript: str = ""


class AgentResult(BaseModel):
    """Result returned by any document agent."""

    metadata: DocumentMetadata
    elapsed_s: float = 0.0
    pages: int = 0
    chars: int = 0
    ocr_method: str = "vision"  # "vision" | "native"


class BaseDocumentAgent(ABC):
    """Abstract base class for all document processing agents."""

    @abstractmethod
    async def process(self, file_path: str, existing_hints: dict) -> AgentResult:
        """
        Process a document file and return extracted metadata.

        Args:
            file_path: Absolute path to the document file on disk.
            existing_hints: Dict with optional keys 'title', 'date', 'correspondent'
                            pre-populated from Paperless for the LLM's context.

        Returns:
            AgentResult with populated metadata and telemetry fields.
        """
        ...
