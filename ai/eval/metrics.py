"""
Pure scoring functions for document intelligence evaluation.

No Phoenix dependency — produces scalar scores that can be logged anywhere.
"""

from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from typing import Optional


# Suffixes to strip for correspondent normalization
_CORP_SUFFIXES = re.compile(
    r"\b(inc|ltd|llc|ag|gmbh|corp|co|company|limited|incorporated)\b",
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Lowercase, remove dots (collapses abbreviations), strip corp suffixes."""
    n = name.lower().strip()
    # Remove dots without inserting spaces so "U.S.A." → "usa", not "u s a"
    n = n.replace(".", "")
    n = _CORP_SUFFIXES.sub("", n)
    # Replace remaining non-word chars with spaces
    n = re.sub(r"[^\w\s]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _token_sort_ratio(s1: str, s2: str) -> float:
    """Sort tokens alphabetically then compute similarity ratio (stdlib only)."""
    t1 = " ".join(sorted(s1.split()))
    t2 = " ".join(sorted(s2.split()))
    return SequenceMatcher(None, t1, t2).ratio()


def score_correspondent(
    expected: Optional[str], actual: Optional[str]
) -> dict:
    """
    Score correspondent extraction quality.

    Returns:
        corr_exact_match: bool — case-insensitive match after normalization
        corr_fuzzy_score: float in [0, 1] — token sort ratio
        corr_expected_null: bool — ground truth was null
        corr_predicted_null: bool — model predicted null
    """
    exp_null = expected is None
    act_null = actual is None

    if exp_null and act_null:
        return {
            "corr_exact_match": True,
            "corr_fuzzy_score": 1.0,
            "corr_expected_null": True,
            "corr_predicted_null": True,
        }
    if exp_null or act_null:
        return {
            "corr_exact_match": False,
            "corr_fuzzy_score": 0.0,
            "corr_expected_null": exp_null,
            "corr_predicted_null": act_null,
        }

    norm_exp = _normalize_name(expected)
    norm_act = _normalize_name(actual)
    return {
        "corr_exact_match": norm_exp == norm_act,
        "corr_fuzzy_score": _token_sort_ratio(norm_exp, norm_act),
        "corr_expected_null": False,
        "corr_predicted_null": False,
    }


def score_date(
    expected: Optional[str], actual: Optional[str]
) -> dict:
    """
    Score date extraction quality.

    Returns:
        date_exact_match: bool
        date_distance_days: int | None
        date_partial_credit: float in [0, 1] — linear decay, zero at 365+ days
        date_null_correct: bool | None
    """
    if expected is None and actual is None:
        return {
            "date_exact_match": True,
            "date_distance_days": 0,
            "date_partial_credit": 1.0,
            "date_null_correct": True,
        }
    if expected is None or actual is None:
        return {
            "date_exact_match": False,
            "date_distance_days": None,
            "date_partial_credit": 0.0,
            "date_null_correct": False,
        }

    try:
        d_exp = date.fromisoformat(expected)
        d_act = date.fromisoformat(actual)
        dist = abs((d_exp - d_act).days)
        partial = max(0.0, 1.0 - dist / 365.0)
        return {
            "date_exact_match": dist == 0,
            "date_distance_days": dist,
            "date_partial_credit": partial,
            "date_null_correct": None,
        }
    except (ValueError, TypeError):
        return {
            "date_exact_match": False,
            "date_distance_days": None,
            "date_partial_credit": 0.0,
            "date_null_correct": None,
        }


def score_title(
    expected_contains: Optional[str], actual: Optional[str]
) -> dict:
    """
    Score title extraction quality against an expected keyword substring.

    Returns:
        title_contains_match: bool | None — None if no expected keyword defined
        title_keyword_overlap: float | None — Jaccard similarity of word sets
    """
    if not expected_contains:
        return {"title_contains_match": None, "title_keyword_overlap": None}

    if not actual:
        return {"title_contains_match": False, "title_keyword_overlap": 0.0}

    contains = expected_contains.lower() in actual.lower()

    exp_words = set(expected_contains.lower().split())
    act_words = set(actual.lower().split())
    union = exp_words | act_words
    jaccard = len(exp_words & act_words) / len(union) if union else 0.0

    return {"title_contains_match": contains, "title_keyword_overlap": jaccard}


def aggregate_scores(row_scores: list[dict]) -> dict:
    """
    Compute experiment-level aggregates from a list of per-row score dicts.

    Each row_score dict is the merged output of score_correspondent,
    score_date, and score_title for one document.
    """
    if not row_scores:
        return {}

    def _mean(values):
        clean = [v for v in values if v is not None]
        return sum(clean) / len(clean) if clean else None

    def _accuracy(bools):
        clean = [v for v in bools if v is not None]
        return sum(clean) / len(clean) if clean else None

    # Correspondent
    corr_exact = _accuracy([r.get("corr_exact_match") for r in row_scores])
    corr_fuzzy = _mean([r.get("corr_fuzzy_score") for r in row_scores])

    # Null handling precision and recall
    # Precision: of entries where model predicted null, what % had ground truth null?
    # Recall: of entries where ground truth is null, what % did model predict null?
    predicted_null = [r for r in row_scores if r.get("corr_predicted_null") is True]
    expected_null = [r for r in row_scores if r.get("corr_expected_null") is True]
    both_null = [r for r in row_scores if r.get("corr_expected_null") and r.get("corr_predicted_null")]

    null_precision = (
        len(both_null) / len(predicted_null) if predicted_null else None
    )
    null_recall = (
        len(both_null) / len(expected_null) if expected_null else None
    )

    # Date
    date_exact = _accuracy([r.get("date_exact_match") for r in row_scores])
    date_partial = _mean([r.get("date_partial_credit") for r in row_scores])

    # Title (only entries with expected_contains)
    title_vals = [r.get("title_contains_match") for r in row_scores if r.get("title_contains_match") is not None]
    title_rate = _accuracy(title_vals) if title_vals else None

    return {
        "correspondent_exact_accuracy": corr_exact,
        "correspondent_fuzzy_mean": corr_fuzzy,
        "null_precision": null_precision,
        "null_recall": null_recall,
        "date_exact_accuracy": date_exact,
        "date_partial_mean": date_partial,
        "title_contains_rate": title_rate,
        "n": len(row_scores),
    }
