"""Compatibility wrapper for the shared Paperless client package."""

from paperless_common import paperless as _paperless
from paperless_common.paperless import PaperlessClient, _raise_for_status

niquests = _paperless.niquests

__all__ = ["PaperlessClient", "_raise_for_status", "niquests"]
