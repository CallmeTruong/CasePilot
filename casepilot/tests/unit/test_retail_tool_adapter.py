"""Executable tool declarations and the isolated Retail compatibility boundary."""

import json
from datetime import date
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError


class LookupArgs(BaseModel):
    key: str = Field(description="Customer lookup key")


class LookupResult(BaseModel):
    key: str
    created: date


class Toolkit:
    def __init__(self):
        self.calls = []

    def use_tool(self, name, **arguments):
        self.calls.append((name, arguments))
        return LookupResult(key=arguments["key"], created=date(2026, 11, 7))


def plain_environment():
    definition = SimpleNamespace(
        name="lookup",
        params=LookupArgs,
        openai_schema={
            "type": "function",
            "function": {"name": "lookup", "description": "Find a customer record"},
        },
    )
    return SimpleNamespace(tools=Toolkit(), get_tools=lambda: [definition])


def test_catalog_invokes_business_tool_with_validated_schema_and_serializes_models():
    from casepilot.integration.tau2.tools import create_executor, describe_tools
    from casepilot.tools.definitions import tool_schema

    env = plain_environment()
    definitions = describe_tools(env.get_tools())
    executor = create_executor(env, definitions)
    (tool,) = definitions
    assert isinstance(executor._tools[tool.name], StructuredTool)
    assert env.tools.calls == []
    schema = tool_schema(tool)["function"]
    assert schema["name"] == "lookup"
    assert schema["description"] == "Find a customer record"
    assert schema["parameters"]["required"] == ["key"]
    assert json.loads(executor.invoke("lookup", {"key": "alice"})) == {
        "key": "alice",
        "created": "2026-11-07",
    }
    assert env.tools.calls == [("lookup", {"key": "alice"})]
    with pytest.raises(ValidationError):
        executor.invoke("lookup", {})
    assert len(env.tools.calls) == 1


def test_backend_declarations_become_neutral_metadata():
    from casepilot.integration.tau2.tools import describe_tools
    from casepilot.tools.definitions import tool_schema

    (definition,) = describe_tools(plain_environment().get_tools())
    assert not hasattr(definition, "invoke")
    schema = tool_schema(definition)["function"]
    assert schema["name"] == "lookup"
    assert schema["description"] == "Find a customer record"
    assert schema["parameters"]["required"] == ["key"]


@pytest.mark.parametrize(
    ("result", "expected"),
    [("alice_id", "alice_id"), (25, '"25"'), ({"total": 25}, '{"total": "25"}')],
)
def test_catalog_preserves_retail_tool_result_encoding(result, expected):
    from casepilot.integration.tau2.tools import create_executor, describe_tools

    env = plain_environment()
    env.tools = SimpleNamespace(use_tool=lambda name, **kwargs: result)
    executor = create_executor(env, describe_tools(env.get_tools()))
    assert executor.invoke("lookup", {"key": "alice"}) == expected


def test_retail_catalog_uses_real_backend_business_semantics(retail_environment):
    from casepilot.integration.tau2.tools import create_executor, describe_tools

    definitions = describe_tools(retail_environment.get_tools())
    tools = {tool.name: tool for tool in definitions}
    executor = create_executor(retail_environment, definitions)
    assert executor.invoke("calculate", {"expression": "12.5 * 2"}) == "25.0"
    assert tools["get_order_details"].args_schema.model_fields["order_id"].is_required()


def test_retail_facade_converts_messages_only_at_guarded_dispatch_boundary():
    from tau2.data_model.message import ToolCall as TauToolCall
    from tau2.data_model.message import ToolMessage as TauToolMessage

    from casepilot.integration.tau2.backend import RetailBackend
    from casepilot.messages import ToolCall, ToolMessage

    class Guard:
        agent = object()
        tools = object()

        def get_response(self, message):
            assert isinstance(message, TauToolCall)
            return TauToolMessage(
                id=message.id,
                role="tool",
                content='"safe"',
                error=True,
                execution_status="rejected",
            )

    backend = RetailBackend(Guard())
    result = backend.execute(ToolCall(id="call-1", name="lookup", arguments={}))
    assert isinstance(result, ToolMessage)
    assert result.id == "call-1"
    assert result.content == '"safe"'
    assert result.error is True
    assert result.execution_status == "rejected"
    assert backend.agent is backend._environment.agent
    assert not hasattr(backend, "tools")


def test_public_facade_tools_cannot_bypass_authorization(retail_environment, tmp_path):
    from langchain_core.tools import ToolException

    from casepilot.agents.agent import Agent
    from casepilot.config import RunConfig
    from casepilot.integration.tau2.backend import RetailBackend
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.integration.tau2.tools import describe_tools

    agent = Agent(
        describe_tools(retail_environment.get_tools()), retail_environment.get_policy(), RunConfig()
    )
    agent.get_init_state()
    guarded = GuardedEnvironment(retail_environment, agent, journal_path=tmp_path / "journal.db")
    try:
        public = {tool.name: tool for tool in RetailBackend(guarded).as_tools()}
        customer = next(iter(retail_environment.tools.db.users))
        with pytest.raises(ToolException):
            public["get_user_details"].invoke({"user_id": customer})
        assert guarded.dispatch_count == 0
    finally:
        guarded.close()
