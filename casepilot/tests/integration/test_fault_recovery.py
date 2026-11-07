import uuid
from types import SimpleNamespace

import pytest

from casepilot.agents.state import CaseState
from casepilot.agents.verifier import PolicyGate
from casepilot.integration.tau2.tools import describe_tools


def setup_operation(environment):
    from tau2.data_model.message import ToolCall

    order = next(o for o in environment.tools.db.orders.values() if o.status == "pending")
    episode = uuid.uuid4().hex
    state = CaseState(
        episode_id=episode, customer_id=order.user_id, owned_order_ids={order.order_id}
    )
    gate = PolicyGate(describe_tools(environment.get_tools()))
    arguments = {"order_id": order.order_id, "reason": "no longer needed"}
    gate.check("cancel_pending_order", arguments, state)
    gate.observe_user("yes", state)
    agent = SimpleNamespace(
        episode_id=episode, current_state=state, engine=SimpleNamespace(gate=gate)
    )
    return (
        agent,
        order,
        ToolCall(id=uuid.uuid4().hex, name="cancel_pending_order", arguments=arguments),
    )


@pytest.mark.parametrize("strategy,success", [("S0", False), ("S1", True), ("S2", True)])
def test_pre_execution_timeout_has_no_duplicate_mutation(
    tmp_path, retail_environment, strategy, success
):
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.reliability.fault_injector import FaultInjector

    agent, order, call = setup_operation(retail_environment)
    wrapper = GuardedEnvironment(
        retail_environment,
        agent,
        recovery=strategy,
        backoff_seconds=0,
        journal_path=tmp_path / "operations.sqlite",
        fault=FaultInjector("F1", call.name),
    )
    try:
        result = wrapper.get_response(call)
        assert result.error is not success
        assert wrapper.dispatch_count == int(success)
        assert order.status == ("cancelled" if success else "pending")
        assert wrapper.fault.events[0]["backend_committed"] is False
    finally:
        wrapper.close()


@pytest.mark.parametrize("strategy,success", [("S0", False), ("S1", False), ("S2", True)])
def test_post_commit_loss_never_replays_mutation(tmp_path, retail_environment, strategy, success):
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.reliability.fault_injector import FaultInjector

    agent, order, call = setup_operation(retail_environment)
    wrapper = GuardedEnvironment(
        retail_environment,
        agent,
        recovery=strategy,
        backoff_seconds=0,
        journal_path=tmp_path / "operations.sqlite",
        fault=FaultInjector("F2", call.name),
    )
    try:
        result = wrapper.get_response(call)
        assert result.error is not success
        assert order.status == "cancelled" and wrapper.dispatch_count == 1
        assert sum(wrapper.mutation_commits.values()) == 1
        assert wrapper.fault.events[0]["backend_committed"] is True
        assert wrapper.journal.get(agent.episode_id, call.id)["state"] == "committed"
    finally:
        wrapper.close()


def test_stale_state_requires_fresh_read_and_new_confirmation(tmp_path, retail_environment):
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.reliability.fault_injector import FaultInjector

    agent, order, call = setup_operation(retail_environment)
    wrapper = GuardedEnvironment(
        retail_environment,
        agent,
        recovery="S2",
        backoff_seconds=0,
        journal_path=tmp_path / "operations.sqlite",
        fault=FaultInjector("F3", call.name),
    )
    try:
        result = wrapper.get_response(call)
        assert result.error and order.status == "processed"
        assert not wrapper.mutation_commits
        agent.engine.gate.observe_result(
            call.name, call.arguments, result.content, True, agent.current_state
        )
        assert order.order_id in agent.current_state.required_refresh_order_ids
        verdict = agent.engine.gate.check(call.name, call.arguments, agent.current_state)
        assert not verdict.allowed and verdict.category == "state_conflict"
    finally:
        wrapper.close()
