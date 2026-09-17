"""Small data models for correspondent cleanup clustering."""

from dataclasses import dataclass, field

from paperless_ai.correspondent.resolver import SimilarityComponents


@dataclass
class CorrespondentCluster:
    """In-memory group of existing Paperless correspondents."""

    member_ids: list[int]
    member_names: list[str]
    document_ids: list[int] = field(default_factory=list)
    sample_titles: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[int, ...]:
        """Return stable identity for this exact cluster membership."""
        return tuple(sorted(self.member_ids))


@dataclass(frozen=True)
class ClusterCandidate:
    """Deterministic evidence for a plausible pair of clusters."""

    left_key: tuple[int, ...]
    right_key: tuple[int, ...]
    score: float
    components: SimilarityComponents
    evidence_left_name: str
    evidence_right_name: str
