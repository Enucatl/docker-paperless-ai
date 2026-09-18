"""Unit tests for the single-request Jev metadata evaluator."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from paperless_ai.eval.jev_evaluator import JevMetadataEvaluator


@pytest.mark.asyncio
async def test_evaluate_uses_one_request_for_all_fields():
    """One document produces one request containing three shared questions."""
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-test",
        nouls={
            "date": SimpleNamespace(noul=0.97, confidence=0.86),
            "correspondent": SimpleNamespace(noul=0.84, confidence=0.75),
            "title": SimpleNamespace(noul=0.91, confidence=0.92),
        },
    )
    evaluator = JevMetadataEvaluator(client, "jev-test")

    result = await evaluator.evaluate(
        document_context="exact metadata context",
        title="Invoice",
        date="2024-01-15",
        correspondent="Acme",
    )

    assert client.system_one.call_count == 1
    request = client.system_one.call_args.kwargs
    assert request["state"] == {
        "document": "exact metadata context",
        "predicted_metadata": {
            "title": "Invoice",
            "date": "2024-01-15",
            "correspondent": "Acme",
        },
    }
    assert set(request["questions"]) == {"date", "correspondent", "title"}
    assert all(question.instructions for question in request["questions"].values())
    assert result.date_score == 0.97
    assert result.correspondent_score == 0.84
    assert result.title_score == 0.91
    assert result.date_confidence == 0.86
    assert result.correspondent_confidence == 0.75
    assert result.title_confidence == 0.92
    assert result.aggregate_score == pytest.approx((0.97 + 0.84 + 0.91) / 3)
    assert result.model == "jev-test"


@pytest.mark.asyncio
async def test_noul_confidence_falls_back_to_its_most_likely_outcome():
    """Noul probabilities remain useful with SDKs that omit confidence."""
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-test",
        nouls={
            "date": SimpleNamespace(noul=0.97),
            "correspondent": SimpleNamespace(noul=0.40),
            "title": SimpleNamespace(noul=0.50),
        },
    )
    evaluator = JevMetadataEvaluator(client, "jev-test")

    result = await evaluator.evaluate(
        document_context="context",
        title="Title",
        date="2024-01-01",
        correspondent="Correspondent",
    )

    assert result.date_confidence == 0.97
    assert result.correspondent_confidence == 0.60
    assert result.title_confidence == 0.50


@pytest.mark.asyncio
async def test_null_predictions_are_passed_to_jev():
    """Null metadata is judged by Jev instead of being rejected in Python."""
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-test",
        nouls={
            "date": SimpleNamespace(noul=0.8),
            "correspondent": SimpleNamespace(noul=0.7),
            "title": SimpleNamespace(noul=0.6),
        },
    )
    evaluator = JevMetadataEvaluator(client, "jev-test")

    await evaluator.evaluate(
        document_context="document without useful metadata",
        title=None,
        date=None,
        correspondent=None,
    )

    prediction = client.system_one.call_args.kwargs["state"]["predicted_metadata"]
    assert prediction == {"title": None, "date": None, "correspondent": None}


@pytest.mark.asyncio
async def test_missing_answer_is_a_single_evaluation_failure():
    """A malformed response raises once for the caller to handle conservatively."""
    client = MagicMock()
    client.system_one.return_value = SimpleNamespace(
        model="jev-test",
        nouls={"date": SimpleNamespace(noul=1.0)},
    )
    evaluator = JevMetadataEvaluator(client, "jev-test")

    with pytest.raises(KeyError):
        await evaluator.evaluate(
            document_context="context",
            title="Title",
            date="2024-01-01",
            correspondent="Correspondent",
        )

    assert client.system_one.call_count == 1
