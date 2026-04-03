"""Tests for eval/metrics.py scoring functions."""

import pytest
from paperless_ai.eval.metrics import score_correspondent, score_date, score_title, aggregate_scores


class TestCorrespondent:
    def test_both_null(self):
        result = score_correspondent(None, None)
        assert result["corr_exact_match"] is True
        assert result["corr_fuzzy_score"] == 1.0
        assert result["corr_expected_null"] is True
        assert result["corr_predicted_null"] is True

    def test_expected_null_actual_value(self):
        result = score_correspondent(None, "Some Company")
        assert result["corr_exact_match"] is False
        assert result["corr_fuzzy_score"] == 0.0
        assert result["corr_expected_null"] is True
        assert result["corr_predicted_null"] is False

    def test_expected_value_actual_null(self):
        result = score_correspondent("Some Company", None)
        assert result["corr_exact_match"] is False
        assert result["corr_fuzzy_score"] == 0.0
        assert result["corr_expected_null"] is False
        assert result["corr_predicted_null"] is True

    def test_exact_match_case_insensitive(self):
        result = score_correspondent("ACME Corp", "acme corp")
        assert result["corr_exact_match"] is True
        assert result["corr_fuzzy_score"] == 1.0
        assert result["corr_expected_null"] is False
        assert result["corr_predicted_null"] is False

    def test_exact_match_with_suffix_removal(self):
        result = score_correspondent("Philip Morris Inc.", "Philip Morris")
        assert result["corr_exact_match"] is True
        assert result["corr_expected_null"] is False
        assert result["corr_predicted_null"] is False

    def test_abbreviation_normalization(self):
        """Test that U.S.A. and USA normalize to the same string."""
        result = score_correspondent("Philip Morris U.S.A.", "Philip Morris USA")
        assert result["corr_exact_match"] is True
        assert result["corr_fuzzy_score"] == 1.0

    def test_fuzzy_match_word_reorder(self):
        result = score_correspondent("Philip Morris U.S.A.", "USA Philip Morris")
        assert result["corr_exact_match"] is False
        assert result["corr_fuzzy_score"] > 0.8  # High fuzzy score despite reorder
        assert result["corr_expected_null"] is False
        assert result["corr_predicted_null"] is False

    def test_partial_match(self):
        result = score_correspondent("Brown & Williamson", "Brown & Williamson Tobacco")
        assert result["corr_exact_match"] is False
        assert 0.5 < result["corr_fuzzy_score"] < 1.0


class TestDate:
    def test_both_null(self):
        result = score_date(None, None)
        assert result["date_exact_match"] is True
        assert result["date_distance_days"] == 0
        assert result["date_partial_credit"] == 1.0
        assert result["date_null_correct"] is True

    def test_expected_null_actual_value(self):
        result = score_date(None, "2024-03-01")
        assert result["date_exact_match"] is False
        assert result["date_distance_days"] is None
        assert result["date_partial_credit"] == 0.0
        assert result["date_null_correct"] is False

    def test_exact_match(self):
        result = score_date("1983-01-13", "1983-01-13")
        assert result["date_exact_match"] is True
        assert result["date_distance_days"] == 0
        assert result["date_partial_credit"] == 1.0
        assert result["date_null_correct"] is None

    def test_close_match_one_day(self):
        result = score_date("1983-01-13", "1983-01-14")
        assert result["date_exact_match"] is False
        assert result["date_distance_days"] == 1
        assert 0.99 < result["date_partial_credit"] < 1.0

    def test_one_year_difference(self):
        result = score_date("1983-01-13", "1984-01-13")
        assert result["date_exact_match"] is False
        assert result["date_distance_days"] == 365
        assert result["date_partial_credit"] == 0.0

    def test_half_year_difference(self):
        result = score_date("2024-01-01", "2024-07-02")  # 183 days (2024 is a leap year)
        assert result["date_exact_match"] is False
        assert result["date_distance_days"] == 183
        assert 0.45 < result["date_partial_credit"] < 0.55

    def test_invalid_date_string(self):
        result = score_date("2024-13-01", "2024-01-01")  # Invalid month
        assert result["date_exact_match"] is False
        assert result["date_distance_days"] is None
        assert result["date_partial_credit"] == 0.0


class TestTitle:
    def test_no_expected_contains(self):
        result = score_title(None, "Some Title")
        assert result["title_contains_match"] is None
        assert result["title_keyword_overlap"] is None

    def test_empty_expected_contains(self):
        result = score_title("", "Some Title")
        assert result["title_contains_match"] is None
        assert result["title_keyword_overlap"] is None

    def test_expected_no_actual(self):
        result = score_title("Invoice", None)
        assert result["title_contains_match"] is False
        assert result["title_keyword_overlap"] == 0.0

    def test_substring_match(self):
        result = score_title("Invoice", "Invoice #12345")
        assert result["title_contains_match"] is True
        assert result["title_keyword_overlap"] >= 0.5

    def test_no_substring_match(self):
        result = score_title("Memo", "Invoice #12345")
        assert result["title_contains_match"] is False
        assert result["title_keyword_overlap"] == 0.0

    def test_case_insensitive_substring(self):
        result = score_title("invoice", "INVOICE #12345")
        assert result["title_contains_match"] is True


class TestAggregates:
    def test_perfect_scores(self):
        rows = [
            {
                "corr_exact_match": True,
                "corr_fuzzy_score": 1.0,
                "corr_expected_null": False,
                "corr_predicted_null": False,
                "date_exact_match": True,
                "date_partial_credit": 1.0,
                "title_contains_match": True,
                "title_keyword_overlap": 1.0,
            }
        ]
        agg = aggregate_scores(rows)
        assert agg["correspondent_exact_accuracy"] == 1.0
        assert agg["correspondent_fuzzy_mean"] == 1.0
        assert agg["null_precision"] is None  # No predicted nulls
        assert agg["null_recall"] is None  # No expected nulls
        assert agg["date_exact_accuracy"] == 1.0
        assert agg["date_partial_mean"] == 1.0
        assert agg["title_contains_rate"] == 1.0
        assert agg["n"] == 1

    def test_mixed_scores(self):
        rows = [
            {
                "corr_exact_match": True,
                "corr_fuzzy_score": 1.0,
                "corr_expected_null": False,
                "corr_predicted_null": False,
                "date_exact_match": False,
                "date_partial_credit": 0.5,
                "title_contains_match": False,
                "title_keyword_overlap": 0.0,
            },
            {
                "corr_exact_match": False,
                "corr_fuzzy_score": 0.8,
                "corr_expected_null": False,
                "corr_predicted_null": False,
                "date_exact_match": True,
                "date_partial_credit": 1.0,
                "title_contains_match": None,
                "title_keyword_overlap": None,
            },
        ]
        agg = aggregate_scores(rows)
        assert agg["correspondent_exact_accuracy"] == 0.5  # 1 out of 2
        assert agg["correspondent_fuzzy_mean"] == 0.9  # (1.0 + 0.8) / 2
        assert agg["date_exact_accuracy"] == 0.5  # 1 out of 2
        assert agg["date_partial_mean"] == 0.75  # (0.5 + 1.0) / 2
        assert agg["title_contains_rate"] == 0.0  # 0 out of 1 (second row None skipped)
        assert agg["n"] == 2

    def test_null_precision_and_recall(self):
        """Test null precision and recall calculation."""
        rows = [
            # True positive: both null
            {"corr_expected_null": True, "corr_predicted_null": True, "corr_exact_match": True, "corr_fuzzy_score": 1.0},
            # True positive: both null
            {"corr_expected_null": True, "corr_predicted_null": True, "corr_exact_match": True, "corr_fuzzy_score": 1.0},
            # False negative: expected null but predicted non-null
            {"corr_expected_null": True, "corr_predicted_null": False, "corr_exact_match": False, "corr_fuzzy_score": 0.0},
            # False positive: predicted null but expected non-null
            {"corr_expected_null": False, "corr_predicted_null": True, "corr_exact_match": False, "corr_fuzzy_score": 0.0},
            # True negative: both non-null (doesn't count)
            {"corr_expected_null": False, "corr_predicted_null": False, "corr_exact_match": True, "corr_fuzzy_score": 1.0},
        ]
        agg = aggregate_scores(rows)
        # Precision = TP / (TP + FP) = 2 / (2 + 1) = 2/3
        assert abs(agg["null_precision"] - 2.0/3.0) < 0.01
        # Recall = TP / (TP + FN) = 2 / (2 + 1) = 2/3
        assert abs(agg["null_recall"] - 2.0/3.0) < 0.01

    def test_empty_input(self):
        agg = aggregate_scores([])
        assert agg == {}
