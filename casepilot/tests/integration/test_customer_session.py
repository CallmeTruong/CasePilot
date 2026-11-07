import uuid

import pytest
from pydantic import SecretStr

from casepilot.config import ProviderSettings, RunConfig


def message(text):
    from tau2.data_model.message import AssistantMessage

    return AssistantMessage(role="assistant", content=text)


def call(name, **arguments):
    from tau2.data_model.message import AssistantMessage, ToolCall

    return AssistantMessage(
        role="assistant", tool_calls=[ToolCall(id=uuid.uuid4().hex, name=name, arguments=arguments)]
    )


def build(env, outputs, tmp_path):
    from casepilot.customer.session import CustomerSession
    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment

    outputs = iter(outputs)
    agent = CasePilotAgent(
        env.get_tools(),
        env.get_policy(),
        RunConfig(track="custom", model="openai/gpt-6-luna"),
        generate=lambda **_: next(outputs),
        customer_instruction="Trả lời bằng tiếng Việt.",
    )
    wrapper = GuardedEnvironment(env, agent, journal_path=tmp_path / (agent.episode_id + ".sqlite"))
    from casepilot.integration.tau2.backend import RetailBackend

    return CustomerSession(
        agent, RetailBackend(wrapper), ProviderSettings(api_key=SecretStr("test-only-key"))
    )


def test_direct_chat_reads_receipts_and_deduplicates_without_simulator(
    retail_environment, tmp_path
):
    env = retail_environment
    customer = next(iter(env.tools.db.users.values()))
    session = build(
        env,
        [
            call("find_user_id_by_email", email=customer.email),
            call("get_user_details", user_id=customer.user_id),
            message("Mình đã tìm thấy đơn của bạn."),
        ],
        tmp_path,
    )
    try:
        session.send("Kiểm tra đơn của tôi. Email: " + customer.email, "event1")
        assert session.state.customer_id == customer.user_id
        assert len(session.receipts) == 2
        assert session.turns[0]["input_tokens"] is None
        assert session.transcript[-1]["content"] == "Mình đã tìm thấy đơn của bạn."
        dispatches = session.environment.dispatch_count
        session.send("Kiểm tra đơn của tôi. Email: " + customer.email, "event1")
        assert session.environment.dispatch_count == dispatches
        assert len(session.transcript) == 2
    finally:
        session.close()
    assert session.closed and session.state.customer_id is None and not session.receipts
    with pytest.raises(ValueError):
        session.send("Tin mới", "event2")


def test_confirmation_is_bound_consumed_once_and_rejection_does_not_write(
    retail_environment, tmp_path
):
    env = retail_environment
    order = next(o for o in env.tools.db.orders.values() if o.status == "pending")
    args = {"order_id": order.order_id, "reason": "no longer needed"}
    session = build(
        env,
        [
            call("cancel_pending_order", **args),
            call("cancel_pending_order", **args),
            message("Đơn hàng đã được hủy."),
        ],
        tmp_path,
    )
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send("Hủy đơn", "request")
        confirmation = session.state.confirmation
        token = session.confirmation_token
        assert confirmation.status == "pending" and order.status == "pending"
        with pytest.raises(ValueError):
            session.decide(True, "wrong-fingerprint", "bad-confirm")
        session.decide(True, token, "confirm")
        assert order.status == "cancelled"
        assert sum(session.environment.mutation_commits.values()) == 1
        session.decide(True, token, "confirm")
        assert sum(session.environment.mutation_commits.values()) == 1
    finally:
        session.close()


def test_exception_after_commit_blocks_blind_resubmission(retail_environment, tmp_path):
    env = retail_environment
    order = next(o for o in env.tools.db.orders.values() if o.status == "pending")
    args = {"order_id": order.order_id, "reason": "no longer needed"}
    session = build(
        env, [call("cancel_pending_order", **args), call("cancel_pending_order", **args)], tmp_path
    )
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send("Hủy đơn", "request")
        session.decide(True, session.confirmation_token, "confirm")
        assert order.status == "cancelled"
        assert session.state.status == "unsafe_unknown"
        assert "test-only-key" not in str(session.transcript)
        with pytest.raises(ValueError):
            session.send("Thử lại", "retry")
        assert sum(session.environment.mutation_commits.values()) == 1
    finally:
        session.close()


def test_customer_instruction_is_opt_in(retail_environment):
    from casepilot.agents.state import CaseState
    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent

    config = RunConfig(model="openai/gpt-6-luna")
    baseline = CasePilotAgent(
        retail_environment.get_tools(), retail_environment.get_policy(), config
    )
    customer = CasePilotAgent(
        retail_environment.get_tools(),
        retail_environment.get_policy(),
        config,
        customer_instruction="CUSTOMER_LANGUAGE_NONCE",
    )
    state = CaseState(episode_id="ep")
    assert "CUSTOMER_LANGUAGE_NONCE" not in baseline.engine.prompt(state)[0].content
    assert "CUSTOMER_LANGUAGE_NONCE" in customer.engine.prompt(state)[0].content


def test_reject_and_identity_change_cannot_approve_old_proposal(retail_environment, tmp_path):
    env = retail_environment
    order = next(o for o in env.tools.db.orders.values() if o.status == "pending")
    args = {"order_id": order.order_id, "reason": "no longer needed"}
    session = build(env, [call("cancel_pending_order", **args)], tmp_path)
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send("Hủy đơn", "request")
        bound = session.confirmation_token
        session.state.customer_id = "different"
        with pytest.raises(ValueError):
            session.decide(True, bound, "stale")
        session.state.customer_id = order.user_id
        session.decide(False, bound, "reject")
        assert order.status == "pending" and session.state.confirmation.status == "invalidated"
        assert not session.environment.mutation_commits
    finally:
        session.close()


def test_free_text_yes_does_not_bypass_customer_confirmation_card(retail_environment, tmp_path):
    env = retail_environment
    order = next(o for o in env.tools.db.orders.values() if o.status == "pending")
    args = {"order_id": order.order_id, "reason": "no longer needed"}
    session = build(
        env,
        [
            call("cancel_pending_order", **args),
            call("cancel_pending_order", **args),
            message("Đã xử lý"),
        ],
        tmp_path,
    )
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send("Hủy đơn", "request")
        session.send("yes", "typed-confirmation")
        assert order.status == "pending"
        assert not session.environment.mutation_commits
        assert session.state.confirmation.status == "pending"
    finally:
        session.close()


def test_identical_reproposal_requires_new_customer_token(retail_environment, tmp_path):
    env = retail_environment
    order = next(o for o in env.tools.db.orders.values() if o.status == "pending")
    args = {"order_id": order.order_id, "reason": "no longer needed"}
    session = build(
        env,
        [
            call("cancel_pending_order", **args),
            call("cancel_pending_order", **args),
            call("cancel_pending_order", **args),
            message("Đã xử lý"),
        ],
        tmp_path,
    )
    session.state.customer_id = order.user_id
    session.state.owned_order_ids = {order.order_id}
    try:
        session.send("Hủy đơn", "request")
        old_token = getattr(session, "confirmation_token", session.state.confirmation.fingerprint)
        session.send("Tôi đổi ý, hãy xem lại yêu cầu hủy đơn này", "edit")
        with pytest.raises(ValueError):
            session.decide(True, old_token, "old-button-click")
        assert order.status == "pending"
    finally:
        session.close()


def test_customer_reference_matches_agent_prompt(retail_environment, tmp_path):
    session = build(retail_environment, [], tmp_path)
    try:
        assert session.state.case_id == session.case_reference
        assert session.case_reference in session.agent.engine.prompt(session.state)[0].content
    finally:
        session.close()
