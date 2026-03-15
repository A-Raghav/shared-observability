"""
OTel provider bootstrap.

Call setup_otel() exactly once during application startup, before any
request handling begins.  It wires up all three signal providers:

    Traces  → BatchSpanProcessor      → OTLPSpanExporter   (→ Collector → Tempo)
    Metrics → PeriodicExportingReader → OTLPMetricExporter (→ Collector → Prometheus)
    Logs    → BatchLogRecordProcessor → OTLPLogExporter    (→ Collector → Loki)

It also:
  - Bridges Python's stdlib `logging` module to the OTel log pipeline,
    so every logging.info/warn/error call is automatically forwarded to Loki
    with the current trace_id/span_id injected (enabling log ↔ trace correlation).
  - Auto-instruments FastAPI (request/response spans) and httpx (outbound HTTP spans
    for DuckDuckGo search calls).
  - Installs W3C TraceContext + Baggage propagators for distributed tracing.
"""

import logging
import os

from opentelemetry import metrics, trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# OTel logs (the _logs namespace is the OTel convention — not a private API)
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

_log = logging.getLogger(__name__)

# Custom resource attribute key for agent framework label
_ATTR_FRAMEWORK = "agent.framework"

_initialized = False  # guard against accidental double-initialization


def setup_otel(
    service_name: str,
    service_version: str = "1.0.0",
    framework: str = "unknown",
    collector_endpoint: str | None = None,
    metrics_export_interval_ms: int = 10_000,
) -> None:
    """
    Bootstrap all three OTel signal providers.

    Args:
        service_name:
            Human-readable service identifier that appears in all three backends.
            E.g. "langgraph-app" or "adk-app".
        service_version:
            Semantic version of the service.
        framework:
            Agent framework label — "langgraph" or "adk".
            Attached to every signal as the `agent.framework` resource attribute,
            enabling cross-framework comparison in Grafana.
        collector_endpoint:
            OTLP gRPC endpoint for the OTel Collector.
            Falls back to the OTEL_EXPORTER_OTLP_ENDPOINT env var, then
            "http://localhost:4317" (the Collector's exposed gRPC port).
        metrics_export_interval_ms:
            How often to push metric batches to the Collector. Default 10 s.
    """
    global _initialized
    if _initialized:
        _log.warning("setup_otel() called more than once — skipping re-initialization")
        return

    endpoint = (
        collector_endpoint
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    )

    # ── Shared Resource ────────────────────────────────────────────────────────
    # The Resource is attached to every span, metric, and log record.
    # It tells the backends which service instance produced the signal.
    resource = Resource.create(
        {
            SERVICE_NAME: service_name,
            SERVICE_VERSION: service_version,
            _ATTR_FRAMEWORK: framework,
            "deployment.environment": os.getenv("ENVIRONMENT", "local-dev"),
        }
    )

    # ── Traces ─────────────────────────────────────────────────────────────────
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=endpoint, insecure=True)
        )
    )
    trace.set_tracer_provider(tracer_provider)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=endpoint, insecure=True),
        export_interval_millis=metrics_export_interval_ms,
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # ── Logs ───────────────────────────────────────────────────────────────────
    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(endpoint=endpoint, insecure=True)
        )
    )
    set_logger_provider(logger_provider)

    # Bridge Python stdlib `logging` → OTel log pipeline.
    # Any logging.info/warn/error call from this point forward is:
    #   1. Converted to an OTel LogRecord
    #   2. Enriched with the active trace_id and span_id (if inside a span)
    #   3. Exported to the Collector → Loki
    # This gives us automatic log ↔ trace correlation without changing any
    # existing logging calls in the apps.
    otel_handler = LoggingHandler(
        level=logging.DEBUG, logger_provider=logger_provider
    )
    logging.getLogger().addHandler(otel_handler)

    # ── Propagators ────────────────────────────────────────────────────────────
    # W3C TraceContext: propagates trace_id/span_id in the `traceparent` header.
    # W3C Baggage: propagates key-value pairs in the `baggage` header.
    # Both are industry standards; GCP / Cloud Trace also speaks TraceContext.
    set_global_textmap(
        CompositePropagator(
            [
                TraceContextTextMapPropagator(),
                W3CBaggagePropagator(),
            ]
        )
    )

    # ── Auto-instrumentation ───────────────────────────────────────────────────
    # FastAPI: automatically creates spans for every HTTP request/response.
    # httpx:   automatically creates spans for every outbound HTTP call
    #          (used by the DuckDuckGo search tool internally).
    FastAPIInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()

    _initialized = True

    _log.info(
        "OTel initialized | service=%s framework=%s endpoint=%s",
        service_name,
        framework,
        endpoint,
    )
