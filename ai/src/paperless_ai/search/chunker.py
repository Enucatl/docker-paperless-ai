"""
Text chunker for splitting OCR transcripts into embeddable segments.

Uses LangChain's RecursiveCharacterTextSplitter with configurable chunk size,
overlap, and separator fallback order.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 256,
) -> list[str]:
    """Split *text* into overlapping recursive character chunks.

    Returns an empty list for blank input.  Single-chunk texts are returned
    as-is without truncation.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=min(overlap, max(chunk_size - 1, 0)),
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
