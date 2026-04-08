"""Compatibility wrapper for the shared telemetry helpers."""

from paperless_common.telemetry import (
    add_litellm_metadata,
    set_span_attributes,
    setup_telemetry,
    start_span,
)

__all__ = [
    "add_litellm_metadata",
    "set_span_attributes",
    "setup_telemetry",
    "start_span",
]
