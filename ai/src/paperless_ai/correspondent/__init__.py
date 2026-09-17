"""Deterministic correspondent resolution utilities."""

from paperless_ai.correspondent.resolver import (
    CorrespondentResolution,
    CorrespondentCandidate,
    CorrespondentResolver,
    NameParts,
    SimilarityComponents,
    normalize_name,
)
from paperless_ai.correspondent.clusters import ClusterCandidate, CorrespondentCluster

__all__ = [
    "CorrespondentResolution",
    "CorrespondentCandidate",
    "CorrespondentResolver",
    "NameParts",
    "SimilarityComponents",
    "normalize_name",
    "ClusterCandidate",
    "CorrespondentCluster",
]
