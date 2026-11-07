import json
from datetime import datetime

import pytest

from casepilot.observability.events import redact
from casepilot.observability.telemetry import TraceContext, TraceSink


def test_recursive_redaction_and_hidden_fields(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-example")
    payload = {
        "API_KEY": "provider-secret-example",
        "nested": [
            {
                "content": (
                    "token provider-secret-example email person@example.com phone +1-555-123-4567"
                )
            },
            {"ground_truth": "hidden answer", "user_instructions": "hidden task", "reward": 1},
        ],
    }
    result = json.dumps(redact(payload))
    for forbidden in [
        "provider-secret-example",
        "person@example.com",
        "+1-555-123-4567",
        "hidden answer",
        "hidden task",
    ]:
        assert forbidden not in result


def test_trace_allowlists_fields_and_correlates(tmp_path):
    path = tmp_path / "traces" / "events.jsonl"
    sink = TraceSink(
        path,
        TraceContext(experiment_id="E1", task_id="0", trial=1, episode_id="episode", run_id="run"),
    )
    sink.emit(
        "tool_call",
        tool_name="get_order_details",
        operation_id="op",
        latency_ms=3,
        arguments={"user_id": "private"},
        result={"address": "private"},
        prompt="private",
        reward=1,
    )
    event = json.loads(path.read_text())
    assert event["episode_id"] == "episode"
    assert event["tool_name"] == "get_order_details"
    assert "arguments" not in event and "result" not in event and "prompt" not in event
    assert "reward" not in event
    assert event["timestamp"].endswith("+00:00")
    assert datetime.fromisoformat(event["timestamp"]).tzinfo is not None


def test_no_custom_event_name_can_leak_secrets(tmp_path):
    sink = TraceSink(
        tmp_path / "events.jsonl",
        TraceContext(experiment_id="E1", task_id="0", trial=1, episode_id="ep", run_id="run"),
    )
    with pytest.raises(ValueError, match="event"):
        sink.emit("secret email person@example.com")
