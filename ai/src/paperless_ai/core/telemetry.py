"""
OpenTelemetry setup for local observability via Arize Phoenix.

Call setup_telemetry() once at startup. It is a no-op when
OTEL_EXPORTER_OTLP_ENDPOINT is not set or packages are unavailable.
"""

import logging
import os

log = logging.getLogger(__name__)

_configured = False


def setup_telemetry(*, service_name: str | None = None, project_name: str | None = None) -> None:
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
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk import trace as trace_sdk
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.litellm import LiteLLMInstrumentor
        from openinference.semconv.resource import ResourceAttributes
    except ImportError as e:
        log.warning("Telemetry packages not available: %s — skipping", e)
        return

    resolved_service_name = service_name or os.environ.get("OTEL_SERVICE_NAME") or "paperless-ai"
    resolved_project_name = project_name or os.environ.get("PHOENIX_PROJECT_NAME") or resolved_service_name
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
