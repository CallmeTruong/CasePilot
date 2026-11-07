"""Explicit metadata export must not capture conversations or ambient configuration."""


def test_ambient_tracing_is_not_opt_in(monkeypatch):
    from casepilot.observability.exporters import langsmith

    monkeypatch.delenv("CASEPILOT_LANGSMITH_TRACING", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setattr(
        langsmith, "Client", lambda **kw: (_ for _ in ()).throw(AssertionError("unexpected client"))
    )
    assert langsmith.export_event("tool_call", {"run_id": "run1"}) is None


def test_export_allowlists_and_redacts_metadata(monkeypatch):
    from casepilot.observability.exporters import langsmith

    calls = []
    options = []
    closed = []

    class Client:
        def __init__(self, **kwargs):
            options.append(kwargs)

        def create_run(self, **kwargs):
            calls.append(kwargs)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(langsmith, "Client", Client)
    monkeypatch.setenv("CASEPILOT_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "private-key-123456")
    identifier = langsmith.export_event(
        "tool_call",
        {
            "run_id": "run1",
            "episode_id": "episode1",
            "input_tokens": 12,
            "status": "private-key-123456",
            "prompt": "secret prompt",
            "arguments": {"secret": "data"},
            "result": "secret result",
            "api_key": "private-key-123456",
            "config": {"api_key": "secret"},
        },
    )
    assert identifier
    assert closed == []
    assert langsmith.export_event("tool_result", {"status": "ok"})
    assert len(options) == 1
    langsmith.close()
    langsmith.close()
    assert closed == [True]
    assert options[0]["hide_inputs"] is True and options[0]["hide_outputs"] is True
    assert options[0]["omit_traced_runtime_info"] is True
    assert calls[0]["inputs"] == {} and calls[0]["outputs"] == {}
    metadata = calls[0]["extra"]["metadata"]
    assert metadata == {
        "run_id": "run1",
        "episode_id": "episode1",
        "input_tokens": 12,
        "status": "[REDACTED]",
    }
    assert "secret" not in repr(calls)
