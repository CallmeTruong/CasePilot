import json

from casepilot.config import RunConfig


def test_planner_executor_handoff_runs_through_same_gate(retail_environment):
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if kwargs["call_name"] == "casepilot_planner":
            return AssistantMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "intent": "read another profile",
                        "required_information": [],
                        "evidence_ids": [],
                        "actions": ["get_user_details"],
                        "requires_confirmation": False,
                        "status": "ready",
                    }
                ),
            )
        return AssistantMessage(
            role="assistant",
            tool_calls=[
                ToolCall(id="unauthorized", name="get_user_details", arguments={"user_id": "other"})
            ],
        )

    config = RunConfig(
        architecture="multi", model="openai/gpt-6-luna", user_model="openai/gpt-6-luna"
    )
    agent = CasePilotAgent(
        retail_environment.get_tools(), retail_environment.get_policy(), config, generate=fake
    )
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="Hello"), agent.get_init_state()
    )
    assert not reply.tool_calls and state.customer_id is None
    assert state.llm_calls == 2 and state.plan["intent"] == "read another profile"
    assert calls[0]["tools"] is None  # planner has no dispatch-capable tools
    assert len(calls[1]["tools"]) == 16


def test_invalid_plan_retries_are_bounded(retail_environment):
    from tau2.data_model.message import AssistantMessage, UserMessage

    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent

    calls = []

    def invalid(**kwargs):
        calls.append(kwargs)
        return AssistantMessage(role="assistant", content="invalid plan")

    config = RunConfig(
        architecture="multi",
        model="openai/gpt-6-luna",
        user_model="openai/gpt-6-luna",
        budgets={"max_replans": 2},
    )
    agent = CasePilotAgent(
        retail_environment.get_tools(), retail_environment.get_policy(), config, generate=invalid
    )
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="Hello"), agent.get_init_state()
    )
    assert len(calls) == 3
    assert all(c["tools"] is None for c in calls)
    assert state.status == "rejected" and not reply.tool_calls
