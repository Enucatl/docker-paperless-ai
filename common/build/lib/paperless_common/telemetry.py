"""
OpenTelemetry helpers shared across services.

The listener package can import these safely without carrying OTEL deps because
the heavier imports happen lazily inside setup_telemetry().
"""

import logging
import os
from contextlib import contextmanager

log = logging.getLogger(__name__)

_configured = False


def add_litellm_metadata(kwargs: dict, **fields: str) -> dict:
    """Merge Paperless-specific metadata into a LiteLLM call kwargs dict."""
    metadata = dict(kwargs.get("metadata") or {})
    paperless_ai = dict(metadata.get("paperless_ai") or {})
    paperless_ai.update({k: v for k, v in fields.items() if v is not None})
    metadata["paperless_ai"] = paperless_ai
    kwargs["metadata"] = metadata
    return kwargs


@contextmanager
def start_span(name: str, **attributes):
    """Start an OTEL span when telemetry is available, otherwise no-op."""
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer("paperless_ai")
    except Exception:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        set_span_attributes(span, **attributes)
        yield span


def set_span_attributes(span, **attributes) -> None:
    """Set span attributes, skipping null values and swallowing OTEL errors."""
    if span is None:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            continue


def setup_telemetry(
    *, service_name: str | None = None, project_name: str | None = None
) -> None:
    """Configure OTEL tracing to export to Arize Phoenix."""
    global _configured
    if _configured:
        return

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        log.info("OTEL_EXPORTER_OTLP_ENDPOINT not set — telemetry disabled")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from openinference.semconv.resource import ResourceAttributes
    except ImportError as exc:
        log.warning("Telemetry packages not available: %s — skipping", exc)
        return

    resolved_service_name = (
        service_name or os.environ.get("OTEL_SERVICE_NAME") or "paperless-ai"
    )
    resolved_project_name = (
        project_name or os.environ.get("PHOENIX_PROJECT_NAME") or resolved_service_name
    )
    resource = Resource.create(
        {
            "service.name": resolved_service_name,
            ResourceAttributes.PROJECT_NAME: resolved_project_name,
        }
    )

    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(tracer_provider)

    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    _configured = True
    log.info(
        "Telemetry configured → %s (service=%s project=%s)",
        endpoint,
        resolved_service_name,
        resolved_project_name,
    )
