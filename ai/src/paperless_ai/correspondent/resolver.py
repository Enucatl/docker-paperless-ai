"""Local, deterministic matching for Paperless correspondents."""

from dataclasses import dataclass
from difflib import SequenceMatcher
import unicodedata
from typing import Any


_NOISE_TOKENS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "miss",
        "dr",
        "prof",
        "professor",
        "pty",
        "ltd",
        "limited",
        "gmbh",
        "ag",
        "inc",
        "incorporated",
        "llc",
        "corp",
        "corporation",
        "plc",
        "sa",
        "bv",
        "nv",
    }
)


@dataclass(frozen=True)
class NameParts:
    """The normalized representations of a correspondent name.

    Attributes:
        normalized_string: Space-separated normalized tokens in source order.
        tokens: Normalized tokens in source order.
        token_set: Unique normalized tokens.
        token_sorted_string: Alphabetically sorted normalized tokens.
    """

    normalized_string: str
    tokens: list[str]
    token_set: frozenset[str]
    token_sorted_string: str


@dataclass(frozen=True)
class SimilarityComponents:
    """Individual deterministic similarity scores for one candidate."""

    character: float
    token_sort: float
    jaccard: float
    containment: float
    containment_score: float

    def to_dict(self) -> dict[str, float]:
        """Return JSON-compatible component scores."""
        return {
            "character": self.character,
            "token_sort": self.token_sort,
            "jaccard": self.jaccard,
            "containment": self.containment,
            "containment_score": self.containment_score,
        }


@dataclass(frozen=True)
class CorrespondentResolution:
    """The selected correspondent action and scoring diagnostics."""

    action: str
    observed_name: str
    correspondent_id: int | None
    correspondent_name: str | None
    score: float
    second_best_score: float | None
    components: SimilarityComponents

    def to_audit_dict(self) -> dict[str, Any]:
        """Return the compact JSON-compatible metadata audit trace."""
        return {
            "version": 1,
            "observed": self.observed_name,
            "action": self.action,
            "correspondent_id": self.correspondent_id,
            "correspondent_name": self.correspondent_name,
            "score": self.score,
            "second_best_score": self.second_best_score,
            "components": self.components.to_dict(),
        }


def normalize_name(name: str) -> NameParts:
    """Normalize a name into generic comparison representations.

    Args:
        name: The observed or stored correspondent name.

    Returns:
        Normalized strings, ordered tokens, and a token set.
    """
    decomposed = unicodedata.normalize("NFKD", name.lower())
    accent_free = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    punctuation_free = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in accent_free
    )
    tokens = [token for token in punctuation_free.split() if token not in _NOISE_TOKENS]
    normalized_string = " ".join(tokens)
    return NameParts(
        normalized_string=normalized_string,
        tokens=tokens,
        token_set=frozenset(tokens),
        token_sorted_string=" ".join(sorted(tokens)),
    )


class CorrespondentResolver:
    """Resolve observed names against a cached collection without network calls."""

    def __init__(self, threshold: float = 0.80):
        """Create a resolver with the minimum score for reuse.

        Args:
            threshold: Score at or above which an existing name is reused.
        """
        self.threshold = threshold

    @staticmethod
    def score_names(observed: NameParts, candidate: NameParts) -> SimilarityComponents:
        """Calculate generic similarity components for two normalized names."""
        character = SequenceMatcher(
            None, observed.normalized_string, candidate.normalized_string
        ).ratio()
        token_sort = SequenceMatcher(
            None, observed.token_sorted_string, candidate.token_sorted_string
        ).ratio()
        intersection = observed.token_set & candidate.token_set
        union = observed.token_set | candidate.token_set
        jaccard = len(intersection) / len(union) if union else 0.0
        shortest = min(len(observed.token_set), len(candidate.token_set))
        containment = len(intersection) / shortest if shortest else 0.0
        adjusted_containment = containment
        if shortest == 1:
            adjusted_containment *= 0.90
        containment_score = 0.85 * adjusted_containment + 0.15 * jaccard
        return SimilarityComponents(
            character=character,
            token_sort=token_sort,
            jaccard=jaccard,
            containment=containment,
            containment_score=containment_score,
        )

    def resolve(
        self, observed_name: str, correspondents: list[dict[str, Any]]
    ) -> CorrespondentResolution:
        """Choose an existing correspondent or report that one should be created.

        Args:
            observed_name: Correspondent text extracted from a document.
            correspondents: Cached Paperless correspondent objects.

        Returns:
            The action, selected candidate, and all winning score diagnostics.
        """
        observed = normalize_name(observed_name)
        if not observed.tokens:
            return CorrespondentResolution(
                action="new",
                observed_name=observed_name,
                correspondent_id=None,
                correspondent_name=None,
                score=0.0,
                second_best_score=None,
                components=SimilarityComponents(0.0, 0.0, 0.0, 0.0, 0.0),
            )
        scored: list[tuple[float, int, str, dict[str, Any], SimilarityComponents]] = []
        for candidate in correspondents:
            name = candidate.get("name")
            candidate_id = candidate.get("id")
            if not isinstance(name, str) or not name.strip() or candidate_id is None:
                continue
            candidate_parts = normalize_name(name)
            if not candidate_parts.tokens:
                continue
            components = self.score_names(observed, candidate_parts)
            score = max(
                components.character,
                components.token_sort,
                components.containment_score,
            )
            scored.append(
                (score, int(candidate_id), name.strip(), candidate, components)
            )

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        if not scored:
            return CorrespondentResolution(
                action="new",
                observed_name=observed_name,
                correspondent_id=None,
                correspondent_name=None,
                score=0.0,
                second_best_score=None,
                components=SimilarityComponents(0.0, 0.0, 0.0, 0.0, 0.0),
            )

        score, candidate_id, name, _, components = scored[0]
        second_best_score = scored[1][0] if len(scored) > 1 else None
        return CorrespondentResolution(
            action="existing" if score >= self.threshold else "new",
            observed_name=observed_name,
            correspondent_id=candidate_id if score >= self.threshold else None,
            correspondent_name=name if score >= self.threshold else None,
            score=score,
            second_best_score=second_best_score,
            components=components,
        )
