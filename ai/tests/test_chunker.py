"""
Unit tests for the recursive text chunker.
"""

from paperless_ai.core.config import AgentConfig
from paperless_ai.search.chunker import chunk_text


def test_empty_string_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_stays_single_chunk():
    text = "Hello world"
    assert chunk_text(text, chunk_size=512, overlap=50) == [text]


def test_recursive_chunker_respects_size_limit():
    text = ("paragraph one\n\n" + ("word " * 80) + "\n\nparagraph two\n\n") * 4
    chunks = chunk_text(text, chunk_size=128, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 128 for chunk in chunks)


def test_recursive_chunker_handles_large_overlap_safely():
    text = "a" * 200
    chunks = chunk_text(text, chunk_size=32, overlap=100)

    assert len(chunks) > 1
    assert all(len(chunk) <= 32 for chunk in chunks)
    assert all(chunk for chunk in chunks)


def test_agent_config_reads_chunk_settings():
    config = AgentConfig.model_validate(
        {
            "paperless_url": "http://paperless",
            "paperless_token": "token",
            "metadata_model": "test-metadata-model",
            "chat_model": "test-chat-model",
            "chunk_size": 512,
            "chunk_overlap": 50,
        }
    )

    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
