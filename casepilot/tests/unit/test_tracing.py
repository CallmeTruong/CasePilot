import json

from casepilot.observability.telemetry import TraceContext, TraceSink


def test_export_receives_only_redacted_metadata(monkeypatch, tmp_path):
    received = []

    def capture(event, record):
        received.append((event, dict(record)))
        return "a" * 32

    monkeypatch.setattr("casepilot.observability.exporters.otlp.export_event", capture)
    sink = TraceSink(
        tmp_path / "trace.jsonl",
        TraceContext(experiment_id="E", task_id="0", trial=1, episode_id="ep", run_id="run"),
    )
    sink.emit(
        "tool_call",
        tool_name="lookup",
        operation_id="op",
        arguments={"secret": "x"},
        user_instructions="hidden",
        status="test@example.com",
    )
    record = json.loads(sink.path.read_text())
    assert record["trace_id"] == "a" * 32
    assert "arguments" not in received[0][1]
    assert "user_instructions" not in record
    assert record["status"] == "[REDACTED]"
