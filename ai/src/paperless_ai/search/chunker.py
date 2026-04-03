"""
Text chunker for splitting OCR transcripts into embeddable segments.

Uses character-based splitting (2 048 chars ≈ 512 tokens for typical prose)
with a configurable overlap so context is preserved across chunk boundaries.
"""


def chunk_text(
    text: str,
    max_chars: int = 2048,
    overlap: int = 256,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *max_chars* characters.

    Returns an empty list for blank input.  Single-chunk texts are returned
    as-is without truncation.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        # Calculate next start position: end - overlap, but ensure we advance at least 1 char.
        # This prevents infinite loops when overlap >= max_chars.
        start = max(end - overlap, start + 1)

    return chunks
