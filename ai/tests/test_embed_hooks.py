"""
Unit tests for paperless_ai.core.hooks — chunk situating for embeddings.

All tests are pure unit tests: no Paperless, Redis, or Qdrant required.
The module-level hook cache is reset before every test via the
``reset_hook_cache`` fixture so tests are fully isolated from each other.
"""

import logging
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

import paperless_ai.core.hooks as hooks_module
from paperless_ai.core.hooks import situate_chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_hook_cache(monkeypatch):
    """Reset the module-level hook cache before every test."""
    monkeypatch.setattr(hooks_module, "_hook_resolved", False)
    monkeypatch.setattr(hooks_module, "_cached_hook", None)


@pytest.fixture
def meta():
    """Meta object with all fields populated."""
    class _Meta:
        title = "Test Invoice"
        correspondent = "Acme Corp"
        document_date = "2024-01-15"
        summary = "Invoice for annual software licensing and server maintenance."

    return _Meta()


@pytest.fixture
def meta_none_fields():
    """Meta object where all fields are None."""
    class _Meta:
        title = None
        correspondent = None
        document_date = None
        summary = None

    return _Meta()


@pytest.fixture
def config():
    """AgentConfig with no situation_model and no EMBED_HOOK_FILE."""
    from paperless_ai.core.config import AgentConfig

    return AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="test-token",
    )


@pytest.fixture
def config_with_situation(config):
    """AgentConfig with situation_model set."""
    return config.model_copy(update={"situation_model": "gemini/test-model"})


# ---------------------------------------------------------------------------
# Tier 1 — static metadata header (default path)
# ---------------------------------------------------------------------------


async def test_tier1_header_contains_all_fields(meta, config):
    results = await situate_chunks(["some chunk text"], "full doc", meta, config)
    assert len(results) == 1
    r = results[0]
    assert "Title: Test Invoice" in r
    assert "Sender: Acme Corp" in r
    assert "Date: 2024-01-15" in r
    assert "Summary: Invoice for annual software licensing" in r
    assert "---" in r
    assert r.endswith("some chunk text")


async def test_tier1_summary_omitted_when_none(meta_none_fields, config):
    results = await situate_chunks(["chunk"], "doc", meta_none_fields, config)
    assert "Summary:" not in results[0]
    assert "Title: Unknown" in results[0]


async def test_tier1_unknown_fallback_for_none_fields(meta_none_fields, config):
    results = await situate_chunks(["chunk"], "doc", meta_none_fields, config)
    r = results[0]
    assert "Title: Unknown" in r
    assert "Sender: Unknown" in r
    assert "Date: Unknown" in r


async def test_tier1_multiple_chunks_all_get_header(meta, config):
    chunks = ["chunk one", "chunk two", "chunk three"]
    results = await situate_chunks(chunks, "doc", meta, config)
    assert len(results) == 3
    for chunk, result in zip(chunks, results):
        assert result.endswith(chunk)
        assert "Title: Test Invoice" in result


async def test_tier1_preserves_chunk_content_verbatim(meta, config):
    long_chunk = "word " * 500
    results = await situate_chunks([long_chunk], "doc", meta, config)
    assert results[0].endswith(long_chunk)


async def test_empty_chunks_returns_empty_list(meta, config):
    assert await situate_chunks([], "doc", meta, config) == []


# ---------------------------------------------------------------------------
# Tier 2 — per-chunk LLM situating
# ---------------------------------------------------------------------------


async def test_tier2_calls_llm_once_per_chunk(monkeypatch, meta, config_with_situation):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Context sentence."
    mock_acompletion = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("litellm.acompletion", mock_acompletion)

    chunks = ["chunk A", "chunk B", "chunk C"]
    results = await situate_chunks(chunks, "full document", meta, config_with_situation)

    assert mock_acompletion.call_count == 3
    assert len(results) == 3
    for result in results:
        assert result.startswith("Context sentence.")
        assert "\n\n" in result


async def test_tier2_context_prepended_to_chunk(monkeypatch, meta, config_with_situation):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "This is the context."
    monkeypatch.setattr("litellm.acompletion", AsyncMock(return_value=mock_response))

    results = await situate_chunks(["my chunk"], "doc", meta, config_with_situation)
    assert results[0] == "This is the context.\n\nmy chunk"


async def test_tier2_context_chars_truncates_full_text(monkeypatch, meta):
    from paperless_ai.core.config import AgentConfig

    cfg = AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="test-token",
        situation_model="gemini/test-model",
        situation_context_chars=10,
    )

    captured_prompts = []

    async def capture_call(**kwargs):
        captured_prompts.append(kwargs["messages"][0]["content"])
        resp = MagicMock()
        resp.choices[0].message.content = "ctx"
        return resp

    monkeypatch.setattr("litellm.acompletion", capture_call)

    full_text = "A" * 100
    await situate_chunks(["chunk"], full_text, meta, cfg)

    assert len(captured_prompts) == 1
    # Only first 10 chars of full_text should appear in the prompt
    assert "A" * 10 in captured_prompts[0]
    assert "A" * 11 not in captured_prompts[0]


# ---------------------------------------------------------------------------
# Custom hook file
# ---------------------------------------------------------------------------


async def test_custom_hook_is_called_with_batch_signature(monkeypatch, tmp_path, meta, config):
    hook_file = tmp_path / "my_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def situate_chunks(chunks, full_text, meta, config):
            return [f"CUSTOM:{c}" for c in chunks]
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    results = await situate_chunks(["hello", "world"], "doc", meta, config)
    assert results == ["CUSTOM:hello", "CUSTOM:world"]


async def test_custom_hook_receives_meta_and_config(monkeypatch, tmp_path, meta, config):
    hook_file = tmp_path / "inspect_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def situate_chunks(chunks, full_text, meta, config):
            return [f"{meta.title}|{c}" for c in chunks]
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    results = await situate_chunks(["body"], "doc", meta, config)
    assert results == ["Test Invoice|body"]


async def test_custom_hook_takes_precedence_over_situation_model(
    monkeypatch, tmp_path, meta, config_with_situation
):
    """Custom hook beats situation_model — explicit override wins."""
    hook_file = tmp_path / "override_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def situate_chunks(chunks, full_text, meta, config):
            return [f"HOOK:{c}" for c in chunks]
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))
    mock_llm = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", mock_llm)

    results = await situate_chunks(["chunk"], "doc", meta, config_with_situation)

    assert results == ["HOOK:chunk"]
    mock_llm.assert_not_called()


async def test_missing_hook_file_falls_back_to_tier1(monkeypatch, caplog, meta, config):
    monkeypatch.setenv("EMBED_HOOK_FILE", "/nonexistent/hook.py")

    with caplog.at_level(logging.WARNING, logger="paperless_ai.core.hooks"):
        results = await situate_chunks(["chunk"], "doc", meta, config)

    assert "Title: Test Invoice" in results[0]
    assert any("does not exist" in r.message for r in caplog.records)


async def test_hook_missing_function_falls_back_to_tier1(monkeypatch, tmp_path, caplog, meta, config):
    hook_file = tmp_path / "no_fn.py"
    hook_file.write_text("# nothing here\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        results = await situate_chunks(["chunk"], "doc", meta, config)

    assert "Title: Test Invoice" in results[0]
    assert any(r.exc_info for r in caplog.records)


async def test_hook_syntax_error_falls_back_to_tier1(monkeypatch, tmp_path, caplog, meta, config):
    hook_file = tmp_path / "bad_syntax.py"
    hook_file.write_text("def situate_chunks(\n  # unclosed\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        results = await situate_chunks(["chunk"], "doc", meta, config)

    assert "Title: Test Invoice" in results[0]


async def test_hook_runtime_error_at_import_falls_back(monkeypatch, tmp_path, caplog, meta, config):
    hook_file = tmp_path / "explodes.py"
    hook_file.write_text("raise RuntimeError('boom')\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        results = await situate_chunks(["chunk"], "doc", meta, config)

    assert "Title: Test Invoice" in results[0]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_hook_file_loaded_only_once(monkeypatch, tmp_path):
    hook_file = tmp_path / "counted.py"
    hook_file.write_text(textwrap.dedent("""\
        async def situate_chunks(chunks, full_text, meta, config):
            return chunks
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    import importlib.util
    real_spec = importlib.util.spec_from_file_location
    call_count = 0

    def counting_spec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_spec(*args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", counting_spec)

    from paperless_ai.core.hooks import _resolve_hook
    _resolve_hook()
    _resolve_hook()
    _resolve_hook()

    assert call_count == 1


def test_no_file_cache_state(monkeypatch):
    monkeypatch.delenv("EMBED_HOOK_FILE", raising=False)

    from paperless_ai.core.hooks import _resolve_hook
    result1 = _resolve_hook()
    result2 = _resolve_hook()

    assert result1 is None
    assert result2 is None
    assert hooks_module._hook_resolved is True
    assert hooks_module._cached_hook is None
