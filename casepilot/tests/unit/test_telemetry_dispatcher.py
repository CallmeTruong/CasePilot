import json
import threading

from casepilot.customer.observability import SessionTrace
from casepilot.observability.telemetry import TraceContext, TraceSink


def test_exporter_failures_preserve_both_local_traces(tmp_path):
    from casepilot.observability.dispatcher import Dispatcher

    def fail(*args):
        raise RuntimeError("private provider error")

    dispatcher = Dispatcher(otlp=fail, langsmith=fail)
    sink = TraceSink(
        tmp_path / "events.jsonl",
        TraceContext(experiment_id="E", task_id="0", trial=1, episode_id="ep", run_id="run"),
        dispatcher=dispatcher,
    )
    rows = []
    session = SessionTrace("ep", rows, dispatcher=dispatcher)
    sink.emit("tool_call", status="ok", arguments="private")
    session.emit("tool_result", status="ok", result="private")
    assert dispatcher.flush(2)
    assert dispatcher.close(2)
    assert json.loads(sink.path.read_text())["status"] == "ok"
    assert rows[0]["status"] == "ok"
    assert dispatcher.diagnostics == {
        "otlp_failures": 2,
        "langsmith_failures": 2,
        "langsmith_dropped": 0,
    }
    assert "private" not in repr(dispatcher.diagnostics)


def test_bounded_queue_nonblocking_privacy_copy_and_shutdown():
    from casepilot.observability.dispatcher import Dispatcher

    entered, release = threading.Event(), threading.Event()
    received = []

    def slow(event, record):
        entered.set()
        release.wait(3)
        received.append(record)

    dispatcher = Dispatcher(otlp=lambda *_: None, langsmith=slow, queue_size=1)
    values = ["safe"]
    try:
        dispatcher.dispatch("tool_call", {"evidence_ids": values, "arguments": "private"})
        assert entered.wait(1)
        dispatcher.dispatch("tool_call", {"status": "person@example.com"})
        dispatcher.dispatch("tool_call", {"status": "dropped"})
        values.append("mutated")
        assert not dispatcher.flush(0.01)
        assert not dispatcher.close(0.01)
        assert dispatcher.diagnostics["langsmith_dropped"] == 1
    finally:
        release.set()
    assert dispatcher.close(2)
    assert received == [{"evidence_ids": ["safe"]}, {"status": "[REDACTED]"}]


def test_ambient_langsmith_does_not_start_worker(monkeypatch):
    from casepilot.observability.dispatcher import Dispatcher
    from casepilot.observability.exporters import langsmith

    monkeypatch.delenv("CASEPILOT_LANGSMITH_TRACING", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(langsmith, "Client", lambda **_: (_ for _ in ()).throw(AssertionError()))
    dispatcher = Dispatcher(otlp=lambda *_: None)
    dispatcher.dispatch("tool_call", {"run_id": "run"})
    assert dispatcher.close(1)
    assert dispatcher.diagnostics["langsmith_failures"] == 0


def test_explicit_cloud_flag_queues_reuses_and_closes_client(monkeypatch):
    from casepilot.observability.dispatcher import Dispatcher
    from casepilot.observability.exporters import langsmith

    clients, calls, closed = [], [], []
    entered, release = threading.Event(), threading.Event()

    class Client:
        def __init__(self, **options):
            clients.append(options)

        def create_run(self, **record):
            entered.set()
            release.wait(3)
            calls.append(record)

        def close(self):
            closed.append(True)

    langsmith.close()
    monkeypatch.setattr(langsmith, "Client", Client)
    monkeypatch.setenv("CASEPILOT_LANGSMITH_TRACING", "true")
    dispatcher = Dispatcher(otlp=lambda *_: None)
    try:
        dispatcher.dispatch("tool_call", {"status": "ok", "prompt": "private"})
        assert entered.wait(1)
        dispatcher.dispatch("tool_result", {"status": "person@example.com"})
        assert not dispatcher.flush(0.01)
    finally:
        release.set()
        assert dispatcher.close(2)
    assert len(clients) == 1
    assert len(calls) == 2
    assert calls[1]["extra"]["metadata"]["status"] == "[REDACTED]"
    assert all(call["inputs"] == {} and call["outputs"] == {} for call in calls)
    assert "private" not in repr(calls)
    assert closed == [True]
    dispatcher.dispatch("tool_call", {"status": "closed"})
    assert len(calls) == 2


def test_otlp_direct_entry_defends_metadata(monkeypatch):
    from casepilot.observability.exporters import otlp

    attributes = {}

    class Span:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def set_attribute(self, key, value):
            attributes[key] = value

        def get_span_context(self):
            return type("Context", (), {"trace_id": 1})()

    class Provider:
        def get_tracer(self, name):
            return self

        def start_as_current_span(self, name):
            return Span()

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4318/v1/traces")
    monkeypatch.setattr(otlp, "_provider", Provider())
    assert (
        otlp.export_event(
            "tool_call",
            {"episode_id": "ep", "status": "person@example.com", "arguments": "private"},
        )
        == "0" * 31 + "1"
    )
    assert attributes == {"casepilot.episode_id": "ep", "casepilot.status": "[REDACTED]"}


def test_shared_lifecycle_api(monkeypatch):
    from casepilot.observability import dispatcher as module

    shared = module.Dispatcher(otlp=lambda *_: None)
    monkeypatch.setattr(module, "_default", shared)
    assert module.flush(0)
    assert module.diagnostics() == {
        "otlp_failures": 0,
        "langsmith_failures": 0,
        "langsmith_dropped": 0,
    }
    assert module.close(0)
