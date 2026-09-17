"""Deterministic correspondent resolution utilities."""

from paperless_ai.correspondent.resolver import (
    CorrespondentResolution,
    CorrespondentResolver,
    NameParts,
    SimilarityComponents,
    normalize_name,
)

__all__ = [
    "CorrespondentResolution",
    "CorrespondentResolver",
    "NameParts",
    "SimilarityComponents",
    "normalize_name",
]
