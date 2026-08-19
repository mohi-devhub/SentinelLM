"""OpenTelemetry distributed tracing — opt-in via OTEL_EXPORTER_OTLP_ENDPOINT.

`tracer` is a module-level OTel `_ProxyTracer`: it is safe to import and use
at any point (including before configure_tracing() runs), and automatically
starts emitting real spans once a real TracerProvider is installed. Until
then every span it creates is a documented OTel no-op — no conditional logic
is needed at any call site, and a deployment that never sets the OTLP
endpoint pays zero cost.
"""

from __future__ import annotations

import logging

from opentelemetry import trace

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("sentinel")


def configure_tracing(app, otel_endpoint: str, otel_headers: str, service_name: str) -> None:
    """Install a real TracerProvider + OTLP exporter and instrument `app`.

    A no-op when otel_endpoint is empty — the default no-op provider (and
    `tracer` above) is left exactly as-is.
    """
    if not otel_endpoint:
        logger.info("tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT unset)")
        return

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    headers = (
        dict(pair.split("=", 1) for pair in otel_headers.split(",") if "=" in pair)
        if otel_headers
        else None
    )

    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))
    exporter = OTLPSpanExporter(endpoint=otel_endpoint, headers=headers)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    logger.info("tracing enabled -> %s", otel_endpoint)


def shutdown_tracing() -> None:
    """Flush and close the exporter. Safe to call even when never configured."""
    shutdown = getattr(trace.get_tracer_provider(), "shutdown", None)
    if shutdown is not None:
        shutdown()
