"""Tests for deterministic local correspondent resolution."""

import pytest

from paperless_ai.correspondent.resolver import CorrespondentResolver, normalize_name


def test_normalize_name_removes_accents_punctuation_and_safe_tokens():
    """Normalization keeps meaningful tokens while removing globally safe noise."""
    normalized = normalize_name("Dr. M\u00e4tteo Abis, Pty. Ltd.")

    assert normalized.normalized_string == "matteo abis"
    assert normalized.tokens == ["matteo", "abis"]
    assert normalized.token_set == {"matteo", "abis"}
    assert normalized.token_sorted_string == "abis matteo"


def test_token_order_invariance_resolves_matteo_abis():
    """Titles and reversed person-name order reuse the cached correspondent."""
    resolver = CorrespondentResolver()
    correspondents = [{"id": 11, "name": "Matteo Abis"}]

    titled = resolver.resolve("Dr Matteo Abis", correspondents)
    reversed_name = resolver.resolve("Abis Matteo", correspondents)

    assert titled.action == "existing"
    assert titled.score == pytest.approx(1.0)
    assert reversed_name.action == "existing"
    assert reversed_name.components.token_sort == pytest.approx(1.0)
    assert reversed_name.score == pytest.approx(1.0)


def test_similarity_components_and_one_token_containment_penalty():
    """One-token containment receives the specified 0.90 multiplier."""
    resolver = CorrespondentResolver()
    components = resolver.score_names(
        normalize_name("Zurich"), normalize_name("Zurich Insurance Group")
    )

    assert components.character == pytest.approx(3 / 7)
    assert components.token_sort == pytest.approx(3 / 7)
    assert components.jaccard == pytest.approx(1 / 3)
    assert components.containment == pytest.approx(1.0)
    assert components.containment_score == pytest.approx(0.85 * 0.90 + 0.15 / 3)


@pytest.mark.parametrize(
    ("existing", "observed"),
    [
        ("Jetstar", "Jetstar Airways Pty Ltd."),
        ("Virgin Australia", "Virgin Australia Itinerary"),
    ],
)
def test_generic_containment_reuses_existing_correspondent(existing, observed):
    """Meaningful extra tokens still allow generic containment matching."""
    resolution = CorrespondentResolver().resolve(
        observed, [{"id": 3, "name": existing}]
    )

    assert resolution.action == "existing"
    assert resolution.correspondent_id == 3
    assert resolution.score >= 0.80
    assert resolution.components.containment == pytest.approx(1.0)


def test_threshold_controls_reuse_vs_create():
    """The configured threshold is the sole runtime reuse decision."""
    correspondents = [{"id": 3, "name": "Jetstar"}]

    assert (
        CorrespondentResolver(0.84)
        .resolve("Jetstar Airways Pty Ltd.", correspondents)
        .action
        == "existing"
    )
    assert (
        CorrespondentResolver(0.85)
        .resolve("Jetstar Airways Pty Ltd.", correspondents)
        .action
        == "new"
    )


def test_best_candidate_selection_threshold_and_audit_trace():
    """The highest local score determines reuse and records diagnostics."""
    resolver = CorrespondentResolver(threshold=0.90)
    resolution = resolver.resolve(
        "Abis Matteo",
        [{"id": 9, "name": "Mario Rossi"}, {"id": 2, "name": "Matteo Abis"}],
    )

    assert resolution.action == "existing"
    assert resolution.correspondent_id == 2
    assert resolution.second_best_score is not None
    audit = resolution.to_audit_dict()
    assert audit["version"] == 1
    assert audit["action"] == "existing"
    assert audit["components"]["token_sort"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "observed", ["Completely Unrelated Sender", "Northwind Traders"]
)
def test_clearly_different_names_create_when_below_threshold(observed):
    """Unrelated names do not pass the configured reuse threshold."""
    resolution = CorrespondentResolver().resolve(
        observed, [{"id": 4, "name": "Matteo Abis"}, {"id": 5, "name": "Jetstar"}]
    )

    assert resolution.action == "new"
    assert resolution.score < 0.90
    assert resolution.correspondent_id is None
