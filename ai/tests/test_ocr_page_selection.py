"""
Unit tests for the OCR page-selection logic (_select_ocr_pages).

All tests are pure unit tests — no Paperless, Redis, Qdrant, or LiteLLM.
"""

import pytest

from paperless_ai.agents.smart_graph_agent import _select_ocr_pages
from paperless_ai.core.config import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(threshold: int, first: int, last: int) -> AgentConfig:
    return AgentConfig(
        paperless_url="http://localhost",
        paperless_token="t",
        metadata_model="test-metadata-model",
        chat_model="test-chat-model",
        ocr_page_limit_threshold=threshold,
        ocr_first_pages=first,
        ocr_last_pages=last,
    )


# ---------------------------------------------------------------------------
# Below threshold — all pages selected
# ---------------------------------------------------------------------------


def test_short_doc_all_pages_selected():
    """Documents at or below the threshold get every page."""
    cfg = _cfg(threshold=40, first=20, last=20)
    assert _select_ocr_pages(1, cfg) == [0]
    assert _select_ocr_pages(40, cfg) == list(range(40))


def test_exactly_at_threshold_all_pages():
    """total_pages == threshold → all pages (boundary is inclusive)."""
    cfg = _cfg(threshold=10, first=4, last=1)
    assert _select_ocr_pages(10, cfg) == list(range(10))


# ---------------------------------------------------------------------------
# Above threshold — first N + last M selected
# ---------------------------------------------------------------------------


def test_default_config_long_doc():
    """Default config: 41-page doc → first 20 + last 20 = pages 0–19 and 21–40."""
    cfg = _cfg(threshold=40, first=20, last=20)
    result = _select_ocr_pages(41, cfg)
    # last 20 of 41 pages = indices 21..40 (0-based)
    assert result == list(range(20)) + list(range(21, 41))


def test_original_spec_first4_last1():
    """First 4 + last 1 on an 11-page doc reproduces the original spec example."""
    cfg = _cfg(threshold=10, first=4, last=1)
    result = _select_ocr_pages(11, cfg)
    assert result == [0, 1, 2, 3, 10]


def test_indices_are_sorted():
    """Returned indices are always in ascending order."""
    cfg = _cfg(threshold=5, first=3, last=3)
    result = _select_ocr_pages(10, cfg)
    assert result == sorted(result)


def test_no_duplicates_when_windows_overlap():
    """When first+last windows overlap, each index appears exactly once."""
    cfg = _cfg(threshold=5, first=4, last=4)
    result = _select_ocr_pages(6, cfg)
    assert len(result) == len(set(result))
    # All 6 pages are covered since 4+4 > 6
    assert result == list(range(6))


def test_no_duplicates_adjacent_windows():
    """Windows that meet exactly at a boundary produce no duplicates."""
    cfg = _cfg(threshold=5, first=3, last=3)
    result = _select_ocr_pages(6, cfg)
    assert result == list(range(6))
    assert len(result) == 6


def test_large_doc_disjoint_windows():
    """On a 300-page doc, first 20 + last 20 are disjoint."""
    cfg = _cfg(threshold=40, first=20, last=20)
    result = _select_ocr_pages(300, cfg)
    assert result == list(range(20)) + list(range(280, 300))
    assert len(result) == 40


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_page_doc():
    """Single-page documents never get filtered."""
    cfg = _cfg(threshold=40, first=20, last=20)
    assert _select_ocr_pages(1, cfg) == [0]


def test_first_pages_exceeds_total():
    """ocr_first_pages larger than total_pages clamps to total_pages."""
    cfg = _cfg(threshold=5, first=100, last=5)
    result = _select_ocr_pages(8, cfg)
    # 8 > threshold=5 so limit applies; first 100 clamped to 8 → all pages
    assert result == list(range(8))


def test_last_pages_exceeds_total():
    """ocr_last_pages larger than total_pages clamps gracefully."""
    cfg = _cfg(threshold=5, first=2, last=100)
    result = _select_ocr_pages(8, cfg)
    assert result == list(range(8))


def test_threshold_zero_always_applies_limit():
    """threshold=0 means the limit always fires, even for 1-page docs."""
    cfg = _cfg(threshold=0, first=1, last=0)
    # 1 > 0, so limit applies; first 1 page = [0]
    assert _select_ocr_pages(1, cfg) == [0]
    # 5 > 0, first 1 + last 0 = [0]
    assert _select_ocr_pages(5, cfg) == [0]


def test_last_pages_zero():
    """ocr_last_pages=0 selects only the first N pages."""
    cfg = _cfg(threshold=5, first=3, last=0)
    result = _select_ocr_pages(10, cfg)
    assert result == [0, 1, 2]


def test_first_pages_zero():
    """ocr_first_pages=0 selects only the last M pages."""
    cfg = _cfg(threshold=5, first=0, last=2)
    result = _select_ocr_pages(10, cfg)
    assert result == [8, 9]


def test_both_zero_selects_nothing():
    """Both zero → empty selection for docs above threshold."""
    cfg = _cfg(threshold=5, first=0, last=0)
    result = _select_ocr_pages(10, cfg)
    assert result == []


def test_very_large_threshold_never_limits():
    """A huge threshold effectively disables the limit."""
    cfg = _cfg(threshold=9999, first=4, last=1)
    assert _select_ocr_pages(500, cfg) == list(range(500))
