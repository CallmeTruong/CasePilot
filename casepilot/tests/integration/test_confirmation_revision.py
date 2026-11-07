import json

from casepilot.agents.state import CaseState
from casepilot.agents.verifier import PolicyGate
from casepilot.config import RunConfig
from casepilot.integration.tau2.tools import describe_tools


def pending_change(env):
    customer = next(iter(env.tools.db.users.values()))
    state = CaseState(episode_id="confirmation-revision", customer_id=customer.user_id)
    gate = PolicyGate(describe_tools(env.get_tools()))
    args = {
        "user_id": customer.user_id,
        "address1": "Test",
        "address2": "",
        "city": "Test",
        "state": "CA",
        "country": "USA",
        "zip": "90001",
    }
    gate.check("modify_user_address", args, state)
    return gate, state, args


def test_new_user_turn_revokes_unconsumed_confirmation(retail_environment):
    gate, state, args = pending_change(retail_environment)
    gate.observe_user("yes", state)
    assert gate.check("modify_user_address", args, state).allowed
    gate.observe_user("I changed my mind", state)
    assert state.confirmation.status == "invalidated"
    assert not gate.check("modify_user_address", args, state).allowed


def test_material_replan_invalidates_pending_confirmation(retail_environment):
    from tau2.data_model.message import AssistantMessage

    from casepilot.agents.graph import MultiAgentGraph

    _gate, state, _args = pending_change(retail_environment)
    old = {
        "intent": "address",
        "actions": ["modify address"],
        "requires_confirmation": True,
        "required_information": [],
        "evidence_ids": [],
        "status": "ready",
    }
    state.plan = old

    def propose(**_kwargs):
        return AssistantMessage(
            role="assistant", content=json.dumps(old | {"actions": ["read profile"]})
        )

    graph = MultiAgentGraph(
        retail_environment.get_tools(),
        retail_environment.get_policy(),
        RunConfig(architecture="multi", model="openai/gpt-6-luna"),
        generate=propose,
    )
    graph.plan({"case": state})
    assert state.plan_revision == 1
    assert state.confirmation.status == "invalidated"


def test_confirmed_plan_is_not_rewritten_before_execution(retail_environment):
    from casepilot.agents.graph import MultiAgentGraph

    gate, state, _args = pending_change(retail_environment)
    state.plan = {"intent": "address"}
    gate.observe_user("yes", state)

    def forbidden(**_kwargs):
        raise AssertionError("Confirmed plan must be retained until execution")

    graph = MultiAgentGraph(
        retail_environment.get_tools(),
        retail_environment.get_policy(),
        RunConfig(architecture="multi", model="openai/gpt-6-luna"),
        generate=forbidden,
    )
    graph.plan({"case": state})
    assert state.confirmation.status == "confirmed"
    state.graph_steps = graph.config.budgets.max_graph_steps
    graph.plan({"case": state})
    assert state.status == "exhausted"
