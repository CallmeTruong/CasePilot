from copy import deepcopy

from tests.integration.test_customer_session import build, call, message


def test_live_activity_precedes_provider_and_tool_completion(retail_environment, tmp_path):
    user = next(iter(retail_environment.tools.db.users.values()))
    session = build(
        retail_environment,
        [call("find_user_id_by_email", email=user.email), message("Đã xác minh")],
        tmp_path,
    )
    observed = []
    generate = session.agent.engine.generate

    def checked_generate(**kwargs):
        assert observed and observed[-1][-1]["kind"] == "model"
        assert observed[-1][-1]["status"] == "running"
        return generate(**kwargs)

    response = session.environment.execute

    def checked_tool(tool):
        assert observed[-1][-1]["kind"] == "tool"
        assert observed[-1][-1]["status"] == "running"
        return response(tool)

    session.agent.engine.generate = checked_generate
    session.environment.execute = checked_tool
    try:
        session.send(
            "Kiểm tra email " + user.email,
            "one",
            on_activity=lambda rows: observed.append(deepcopy(rows)),
        )
        assert session.activity
        tool = next(row for row in session.activity if row["kind"] == "tool")
        assert tool["name"] == "find_user_id_by_email" and tool["status"] == "success"
        assert tool["arguments"]["email"] == user.email
        assert tool["latency_ms"] > 0 and tool["result"]
        before = len(observed)
        session.send("lặp", "one", on_activity=lambda rows: observed.append(rows))
        assert len(observed) == before
        assert "arguments" not in str(session.traces)
    finally:
        session.close()
    assert not session.activity


def test_provider_failure_finishes_activity_and_discards_callback(retail_environment, tmp_path):
    session = build(retail_environment, [], tmp_path)

    def fail(**kwargs):
        raise TimeoutError("private endpoint detail")

    session.agent.engine.generate = fail
    seen = []
    try:
        session.send("Xin chào", "one", on_activity=lambda rows: seen.append(rows))
        assert session.activity[-1]["status"] == "error"
        assert "private endpoint" not in str(session.activity)
        before = len(seen)
        session.agent.engine.emit("agent_call", status="success")
        assert len(seen) == before
    finally:
        session.close()


def test_blocked_proposal_shows_arguments_but_never_claims_execution(retail_environment, tmp_path):
    session = build(retail_environment, [call("get_user_details", user_id="someone")], tmp_path)
    try:
        session.send("Xem hồ sơ", "one")
        check = next(row for row in session.activity if row["kind"] == "verification")
        assert check["status"] == "blocked"
        assert check["arguments"] == {"user_id": "someone"}
        assert not any(row["kind"] == "tool" for row in session.activity)
    finally:
        session.close()


def test_preview_removes_secrets_and_bounds_nested_payload(monkeypatch):
    from casepilot.customer.activity import preview

    monkeypatch.setenv("OPENAI_API_KEY", "configured-private-key")
    value = {
        "email": "synthetic@example.com",
        "api_key": "not-for-display",
        "nested": {"note": "configured-private-key", "hidden_instructions": "gold-secret"},
        "long": "a" * 30000,
    }
    safe = preview(value)
    assert safe["email"] == "synthetic@example.com"
    assert "not-for-display" not in str(safe) and "gold-secret" not in str(safe)
    assert "configured-private-key" not in str(safe)
    assert len(str(safe)) < 14000 and "truncated" in str(safe)


def test_parallel_proposals_are_visible_as_blocked_without_dispatch(retail_environment, tmp_path):
    first = call("find_user_id_by_email", email="one@example.test")
    second = call("get_user_details", user_id="someone")
    first.tool_calls.extend(second.tool_calls)
    session = build(retail_environment, [first], tmp_path)
    try:
        session.send("Kiểm tra", "one")
        checks = [row for row in session.activity if row["kind"] == "verification"]
        assert len(checks) == 2
        assert {row["name"] for row in checks} == {"find_user_id_by_email", "get_user_details"}
        assert all(row["status"] == "blocked" and row["arguments"] for row in checks)
        assert session.environment.dispatch_count == 0
    finally:
        session.close()


def test_connection_settings_are_never_in_business_previews():
    from casepilot.customer.activity import preview

    safe = preview({"base_url": "https://private-endpoint.test", "endpoint": "private-connection"})
    assert "private-endpoint" not in str(safe) and "private-connection" not in str(safe)
