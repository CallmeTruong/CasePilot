import atexit
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from casepilot.observability.exporters import otlp as tracing
from casepilot.observability.telemetry import TraceContext, TraceSink


def test_actual_otlp_export_has_correlation_and_no_payload(monkeypatch, tmp_path):
    payloads = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payloads.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"http://127.0.0.1:{server.server_port}/v1/traces"
    )
    monkeypatch.setattr(tracing, "_provider", None)
    try:
        sink = TraceSink(
            tmp_path / "trace.jsonl",
            TraceContext(
                experiment_id="E", task_id="0", trial=1, episode_id="episode", run_id="run"
            ),
        )
        sink.emit("handoff", role="planner", status="ready", arguments="do-not-export")
        assert tracing._provider.force_flush(timeout_millis=5000)
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )

        request = ExportTraceServiceRequest.FromString(payloads[0])
        span = request.resource_spans[0].scope_spans[0].spans[0]
        attributes = {a.key: a.value.string_value for a in span.attributes}
        assert attributes["casepilot.episode_id"] == "episode"
        assert attributes["casepilot.role"] == "planner"
        assert b"do-not-export" not in payloads[0]
        assert span.trace_id.hex() == json.loads(sink.path.read_text())["trace_id"]
    finally:
        if tracing._provider:
            atexit.unregister(tracing._provider.shutdown)
            tracing._provider.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
