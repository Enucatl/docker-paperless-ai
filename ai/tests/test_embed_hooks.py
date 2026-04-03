"""
Unit tests for paperless_ai.core.hooks — the dynamic embed hook system.

All tests are pure unit tests: no Paperless, Redis, or Qdrant required.
The module-level hook cache is reset before every test via the
``reset_hook_cache`` fixture so tests are fully isolated from each other.
"""

import asyncio
import logging
import textwrap

import pytest

import paperless_ai.core.hooks as hooks_module
from paperless_ai.core.hooks import default_embed_hook, get_embed_hook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_hook_cache(monkeypatch):
    """Reset the module-level hook cache before every test.

    get_embed_hook() caches its result in _cached_hook / _hook_resolved so the
    file is only parsed once per process. Without this reset each test would
    see whatever the previous test cached.
    """
    monkeypatch.setattr(hooks_module, "_hook_resolved", False)
    monkeypatch.setattr(hooks_module, "_cached_hook", None)


@pytest.fixture
def meta():
    """Minimal duck-typed meta object matching the DocumentMetadata interface."""
    class _Meta:
        title = "Test Invoice"
        correspondent = "Acme Corp"
        document_date = "2024-01-15"

    return _Meta()


@pytest.fixture
def meta_none_fields():
    """Meta object where all fields are None (tests Unknown fallback)."""
    class _Meta:
        title = None
        correspondent = None
        document_date = None

    return _Meta()


@pytest.fixture
def config():
    """Minimal AgentConfig with no EMBED_HOOK_FILE set."""
    from paperless_ai.core.config import AgentConfig

    return AgentConfig(
        paperless_url="http://localhost:8000",
        paperless_token="test-token",
    )


# ---------------------------------------------------------------------------
# default_embed_hook
# ---------------------------------------------------------------------------


async def test_default_hook_prepends_structured_header(meta, config):
    """Default hook produces Title/Sender/Date header followed by the raw chunk."""
    result = await default_embed_hook("some chunk text", meta, config)

    assert result.startswith("Title: Test Invoice\n")
    assert "Sender: Acme Corp\n" in result
    assert "Date: 2024-01-15\n" in result
    assert result.endswith("some chunk text")
    assert "---\n" in result


async def test_default_hook_uses_unknown_for_none_fields(meta_none_fields, config):
    """None metadata fields render as 'Unknown' rather than 'None'."""
    result = await default_embed_hook("chunk", meta_none_fields, config)

    assert "Title: Unknown" in result
    assert "Sender: Unknown" in result
    assert "Date: Unknown" in result
    assert result.endswith("chunk")


async def test_default_hook_preserves_chunk_content(meta, config):
    """Chunk content is appended verbatim — no truncation or modification."""
    long_chunk = "word " * 500
    result = await default_embed_hook(long_chunk, meta, config)
    assert result.endswith(long_chunk)


async def test_default_hook_empty_chunk(meta, config):
    """Empty chunk still gets a header prepended without error."""
    result = await default_embed_hook("", meta, config)
    assert "Title: Test Invoice" in result
    assert result.endswith("---\n")


# ---------------------------------------------------------------------------
# get_embed_hook — environment not set
# ---------------------------------------------------------------------------


async def test_returns_default_when_env_not_set(monkeypatch, meta, config):
    """When EMBED_HOOK_FILE is unset, get_embed_hook returns default_embed_hook."""
    monkeypatch.delenv("EMBED_HOOK_FILE", raising=False)

    hook = get_embed_hook()
    assert hook is default_embed_hook

    # Calling it produces the same output as the default directly
    result = await hook("chunk", meta, config)
    assert "Title: Test Invoice" in result


async def test_returns_default_when_env_empty_string(monkeypatch, meta, config):
    """Empty string EMBED_HOOK_FILE is treated the same as unset."""
    monkeypatch.setenv("EMBED_HOOK_FILE", "")

    hook = get_embed_hook()
    assert hook is default_embed_hook


# ---------------------------------------------------------------------------
# get_embed_hook — file does not exist
# ---------------------------------------------------------------------------


async def test_returns_default_when_file_missing(monkeypatch, caplog, meta, config):
    """Missing file path logs a warning and falls back to the default hook."""
    monkeypatch.setenv("EMBED_HOOK_FILE", "/nonexistent/path/hook.py")

    with caplog.at_level(logging.WARNING, logger="paperless_ai.core.hooks"):
        hook = get_embed_hook()

    assert hook is default_embed_hook
    assert any("does not exist" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# get_embed_hook — valid custom hook file
# ---------------------------------------------------------------------------


async def test_loads_custom_hook_from_file(monkeypatch, tmp_path, meta, config):
    """A valid hook file is imported and its format_chunk_for_embedding is used."""
    hook_file = tmp_path / "my_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def format_chunk_for_embedding(chunk, meta, config):
            return f"CUSTOM:{chunk}"
    """))

    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    hook = get_embed_hook()
    assert hook is not default_embed_hook

    result = await hook("hello", meta, config)
    assert result == "CUSTOM:hello"


async def test_custom_hook_receives_meta_and_config(monkeypatch, tmp_path, meta, config):
    """The custom hook receives both meta and config correctly."""
    hook_file = tmp_path / "inspect_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def format_chunk_for_embedding(chunk, meta, config):
            return f"{meta.title}|{meta.correspondent}|{chunk}"
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    hook = get_embed_hook()
    result = await hook("body", meta, config)
    assert result == "Test Invoice|Acme Corp|body"


async def test_logs_info_on_successful_load(monkeypatch, tmp_path, caplog):
    """A successfully loaded hook is announced in the log at INFO level."""
    hook_file = tmp_path / "ok_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def format_chunk_for_embedding(chunk, meta, config):
            return chunk
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.INFO, logger="paperless_ai.core.hooks"):
        get_embed_hook()

    assert any("Loaded custom embed hook" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# get_embed_hook — fallback on broken hook files
# ---------------------------------------------------------------------------


async def test_falls_back_when_function_missing(monkeypatch, tmp_path, caplog, meta, config):
    """A file that doesn't define format_chunk_for_embedding falls back to default."""
    hook_file = tmp_path / "no_fn_hook.py"
    hook_file.write_text("# no function here\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        hook = get_embed_hook()

    assert hook is default_embed_hook
    # The error is logged with exc_info; check the exception text in the record
    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert error_records, "Expected an ERROR log record"
    assert any(
        "format_chunk_for_embedding" in str(r.exc_info) for r in error_records
    )


async def test_falls_back_on_syntax_error(monkeypatch, tmp_path, caplog, meta, config):
    """A file with a syntax error is caught, logged, and falls back to default."""
    hook_file = tmp_path / "bad_syntax.py"
    hook_file.write_text("def format_chunk_for_embedding(\n  # unclosed\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        hook = get_embed_hook()

    assert hook is default_embed_hook
    # Exception details logged
    assert any(rec.exc_info is not None for rec in caplog.records)


async def test_falls_back_on_runtime_error_at_import(monkeypatch, tmp_path, caplog, meta, config):
    """A file that raises an exception at module level falls back to default."""
    hook_file = tmp_path / "explodes.py"
    hook_file.write_text("raise RuntimeError('boom')\n")
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    with caplog.at_level(logging.ERROR, logger="paperless_ai.core.hooks"):
        hook = get_embed_hook()

    assert hook is default_embed_hook


# ---------------------------------------------------------------------------
# get_embed_hook — caching
# ---------------------------------------------------------------------------


def test_hook_is_resolved_only_once(monkeypatch, tmp_path):
    """get_embed_hook() loads the file exactly once; subsequent calls use the cache."""
    call_count = 0
    original_spec = __import__("importlib.util", fromlist=["spec_from_file_location"]).spec_from_file_location

    hook_file = tmp_path / "counted_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        async def format_chunk_for_embedding(chunk, meta, config):
            return chunk
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    import importlib.util

    real_spec_from_file = importlib.util.spec_from_file_location

    def counting_spec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_spec_from_file(*args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", counting_spec)

    # Call three times — file should be parsed only on the first call
    h1 = get_embed_hook()
    h2 = get_embed_hook()
    h3 = get_embed_hook()

    assert h1 is h2 is h3
    assert call_count == 1, f"spec_from_file_location called {call_count} times, expected 1"


def test_default_is_cached_too(monkeypatch):
    """get_embed_hook() with no file set caches default_embed_hook as well."""
    monkeypatch.delenv("EMBED_HOOK_FILE", raising=False)

    h1 = get_embed_hook()
    h2 = get_embed_hook()

    assert h1 is h2 is default_embed_hook
    assert hooks_module._hook_resolved is True


# ---------------------------------------------------------------------------
# asyncio.gather compatibility (mirrors _embed_and_store usage)
# ---------------------------------------------------------------------------


async def test_default_hook_works_with_asyncio_gather(meta, config):
    """Default hook can be fanned out concurrently via asyncio.gather."""
    chunks = ["chunk one", "chunk two", "chunk three"]
    hook = get_embed_hook()

    results = list(await asyncio.gather(*(hook(c, meta, config) for c in chunks)))

    assert len(results) == 3
    for chunk, result in zip(chunks, results):
        assert result.endswith(chunk)
        assert "Title: Test Invoice" in result


async def test_custom_hook_works_with_asyncio_gather(monkeypatch, tmp_path, meta, config):
    """Custom async hook integrates correctly with the asyncio.gather call pattern."""
    hook_file = tmp_path / "gather_hook.py"
    hook_file.write_text(textwrap.dedent("""\
        import asyncio
        async def format_chunk_for_embedding(chunk, meta, config):
            await asyncio.sleep(0)   # simulate async work
            return f"[{meta.title}] {chunk}"
    """))
    monkeypatch.setenv("EMBED_HOOK_FILE", str(hook_file))

    hook = get_embed_hook()
    chunks = ["alpha", "beta", "gamma"]

    results = list(await asyncio.gather(*(hook(c, meta, config) for c in chunks)))

    assert results == [
        "[Test Invoice] alpha",
        "[Test Invoice] beta",
        "[Test Invoice] gamma",
    ]
