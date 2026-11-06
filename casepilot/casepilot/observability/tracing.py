"""Optional metadata-only OTLP export to a local Phoenix collector."""

import atexit
import os
import threading

_provider = None
_lock = threading.Lock()


def export_event(event, record):
    global _provider
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    if not endpoint:
        return None
    with _lock:
        if _provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            _provider = TracerProvider(resource=Resource.create({"service.name": "casepilot"}))
            _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
            atexit.register(_provider.shutdown)
    tracer = _provider.get_tracer("casepilot.metadata")
    with tracer.start_as_current_span(event) as span:
        for key, value in record.items():
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute("casepilot." + key, value)
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                span.set_attribute("casepilot." + key, value)
        return format(span.get_span_context().trace_id, "032x")
