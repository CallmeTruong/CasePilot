import uuid

import pytest

from casepilot.agents.state import CaseState
from casepilot.integration.tau2.tools import describe_tools

pytestmark = pytest.mark.integration


def agent_for(environment):
    from types import SimpleNamespace

    from casepilot.agents.verifier import PolicyGate

    episode_id = uuid.uuid4().hex
    return SimpleNamespace(
        episode_id=episode_id,
        current_state=CaseState(episode_id=episode_id),
        engine=SimpleNamespace(gate=PolicyGate(describe_tools(environment.get_tools()))),
    )


def test_stdio_native_parity_and_one_backend_dispatch(retail_environment):
    from tau2.data_model.message import ToolCall
    from tau2.domains.retail.environment import get_environment

    from casepilot.integration.tau2.dispatcher import GuardedEnvironment

    native_env, mcp_env = retail_environment, get_environment()
    native_agent, mcp_agent = agent_for(native_env), agent_for(mcp_env)
    native = GuardedEnvironment(native_env, native_agent)
    proxied = GuardedEnvironment(mcp_env, mcp_agent, backend="mcp")
    customer = next(iter(native_env.tools.db.users.values()))
    try:
        first = ToolCall(
            id=uuid.uuid4().hex, name="find_user_id_by_email", arguments={"email": customer.email}
        )
        assert native.get_response(first) == proxied.get_response(first)
        for _wrapper, agent in ((native, native_agent), (proxied, mcp_agent)):
            agent.current_state.customer_id = customer.user_id
            args = {
                "user_id": customer.user_id,
                "address1": "Synthetic MCP street",
                "address2": "",
                "city": "Test",
                "state": "CA",
                "country": "USA",
                "zip": "90001",
            }
            agent.engine.gate.check("modify_user_address", args, agent.current_state)
            agent.engine.gate.observe_user("yes", agent.current_state)
        mutation = ToolCall(id=uuid.uuid4().hex, name="modify_user_address", arguments=args)
        expected = native.get_response(mutation)
        actual = proxied.get_response(mutation)
        assert expected == actual
        assert proxied.get_response(mutation) == actual
        assert native.get_db_hash() == proxied.get_db_hash()
        assert native.dispatch_count == proxied.dispatch_count == 2
        assert proxied.transport_errors == 0
    finally:
        proxied.close()
        native.close()


def test_bridge_episode_mismatch_rejected_without_dispatch(retail_environment):
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment

    wrapper = GuardedEnvironment(retail_environment, agent_for(retail_environment), backend="mcp")
    try:
        result = wrapper.bridge.request({"episode_id": "another", "action": "list_tools"})
        assert result["protocol_error"] == "episode_mismatch"
        assert wrapper.dispatch_count == 0
    finally:
        wrapper.close()


def test_mcp_postcommit_timeout_reconciles_without_second_mutation(retail_environment, tmp_path):
    from tau2.data_model.message import ToolCall

    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.reliability.fault_injector import FaultInjector

    agent = agent_for(retail_environment)
    customer = next(iter(retail_environment.tools.db.users.values()))
    agent.current_state.customer_id = customer.user_id
    args = {
        "user_id": customer.user_id,
        "address1": "MCP fault test",
        "address2": "",
        "city": "Test",
        "state": "CA",
        "country": "USA",
        "zip": "90001",
    }
    agent.engine.gate.check("modify_user_address", args, agent.current_state)
    agent.engine.gate.observe_user("yes", agent.current_state)
    wrapper = GuardedEnvironment(
        retail_environment,
        agent,
        backend="mcp",
        journal_path=tmp_path / "ops.sqlite",
        fault=FaultInjector("F2", "modify_user_address"),
    )
    try:
        call = ToolCall(id=uuid.uuid4().hex, name="modify_user_address", arguments=args)
        result = wrapper.get_response(call)
        assert not result.error
        assert wrapper.get_response(call) == result
        assert wrapper.mutation_commits[call.id] == 1
        assert wrapper.dispatch_count == 1
        assert wrapper.fault.events[0]["backend_committed"] is True
        assert wrapper.recovery_steps == 1
    finally:
        wrapper.close()
