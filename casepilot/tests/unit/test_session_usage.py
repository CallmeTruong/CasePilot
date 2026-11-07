import pytest


def test_turn_observation_counts_real_usage_and_no_secret_fields(retail_environment, tmp_path):
    from tests.integration.test_customer_session import build, message

    reply = message("Xin chào")
    reply.usage = {"prompt_tokens": 12, "completion_tokens": 7}
    session = build(retail_environment, [reply], tmp_path)
    try:
        session.send("Xin chào", "one")
        assert len(session.turns) == 1
        turn = session.turns[0]
        assert turn["input_tokens"] == 12 and turn["output_tokens"] == 7
        assert turn["latency_ms"] > 0
        assert session.traces and all("arguments" not in row for row in session.traces)
        session.agent.engine.emit(
            "tool_call", arguments={"api_key": "not-a-real-secret"}, status="ok"
        )
        assert "not-a-real-secret" not in str(session.traces)
        assert "test-only-key" not in str(session.traces)
        session.send("Xin chào", "one")
        assert len(session.turns) == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 12},
        {"completion_tokens": 7},
        {"prompt_tokens": 12, "completion_tokens": None},
    ],
)
def test_partial_provider_usage_is_unknown(retail_environment, tmp_path, usage):
    from tests.integration.test_customer_session import build, message

    reply = message("Xin chào")
    reply.usage = usage
    session = build(retail_environment, [reply], tmp_path)
    try:
        session.send("Xin chào", "partial")
        assert not session.state.usage_known
        assert session.turns[0]["input_tokens"] is None
        assert session.turns[0]["output_tokens"] is None
    finally:
        session.close()


def test_failed_provider_attempt_records_unknown_usage_and_model_latency(
    retail_environment, tmp_path
):
    from tests.integration.test_customer_session import build

    session = build(retail_environment, [], tmp_path)

    def timeout(**kwargs):
        raise TimeoutError("private provider connection detail")

    session.agent.engine.generate = timeout
    try:
        session.send("Xin chào", "timeout")
        assert session.turns[0]["llm_calls"] == 1
        assert session.turns[0]["input_tokens"] is None
        assert session.turns[0]["output_tokens"] is None
        assert not session.state.usage_known
        attempt = next(row for row in session.traces if row["event"] == "agent_call")
        assert attempt["status"] == "error" and attempt["latency_ms"] > 0
        assert "private provider" not in str(session.traces)
    finally:
        session.close()


def test_complete_turn_usage_remains_visible_after_an_unknown_turn(retail_environment, tmp_path):
    from tests.integration.test_customer_session import build, message

    reply = message("Mình có thể giúp gì?")
    reply.usage = {"prompt_tokens": 12, "completion_tokens": 7}
    session = build(retail_environment, [message("Xin chào"), reply], tmp_path)
    try:
        session.send("Xin chào", "unknown")
        session.send("Kiểm tra đơn", "known")
        assert session.turns[0]["input_tokens"] is None
        assert session.turns[1]["input_tokens"] == 12
        assert session.turns[1]["output_tokens"] == 7
        assert not session.state.usage_known  # The cumulative session total is still incomplete.
    finally:
        session.close()
