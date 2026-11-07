import ast
from pathlib import Path

import pytest
from pydantic import BaseModel


def test_tool_definition_is_inert_and_schema_is_a_copy():
    from casepilot.tools.definitions import ToolDefinition

    class Args(BaseModel):
        order_id: str

    definition = ToolDefinition("lookup", "Read an order", Args)
    assert not hasattr(definition, "invoke")
    schema = definition.openai_schema
    schema["function"]["parameters"]["properties"].clear()
    assert "order_id" in definition.openai_schema["function"]["parameters"]["properties"]
    with pytest.raises((AttributeError, TypeError)):
        definition.name = "changed"


def test_generic_tools_have_no_retail_structure_dependencies():
    for file in ("definitions.py", "executor.py"):
        path = Path("casepilot/tools") / file
        text = path.read_text(encoding="utf-8")
        assert ".params" not in text and ".use_tool" not in text and "tau2" not in text
        ast.parse(text)


def test_retail_facade_exposes_snapshot_without_backend_forwarding(retail_environment, tmp_path):
    from casepilot.agents.agent import Agent
    from casepilot.config import RunConfig
    from casepilot.integration.tau2.backend import RetailBackend
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.integration.tau2.tools import describe_tools

    agent = Agent(describe_tools(retail_environment.get_tools()), "policy", RunConfig())
    agent.get_init_state()
    dispatcher = GuardedEnvironment(retail_environment, agent, journal_path=tmp_path / "ops.db")
    facade = RetailBackend(dispatcher)
    try:
        snapshot = facade.snapshot()
        snapshot["orders"].clear()
        assert facade.snapshot()["orders"]
        assert not hasattr(facade, "tools")
        assert not hasattr(facade, "_native_response")
        assert facade.tool_definitions is dispatcher.tool_definitions
        assert all(not hasattr(tool, "invoke") for tool in facade.tool_definitions)
    finally:
        facade.close()


def test_backend_closes_original_model_even_when_generate_is_wrapped(retail_environment, tmp_path):
    from casepilot.agents.agent import Agent
    from casepilot.config import RunConfig
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.integration.tau2.tools import describe_tools

    class Model:
        closed = 0

        def __call__(self, **kwargs):
            raise AssertionError("No inference expected")

        def close(self):
            self.closed += 1

    model = Model()
    agent = Agent(
        describe_tools(retail_environment.get_tools()), "policy", RunConfig(), generate=model
    )
    agent.get_init_state()
    agent.engine.generate = lambda **kwargs: model(**kwargs)
    backend = GuardedEnvironment(retail_environment, agent, journal_path=tmp_path / "ops.db")
    backend.close()
    backend.close()
    assert model.closed == 1


def test_runtime_emitters_use_the_shared_event_vocabulary():
    from casepilot.observability.events import EVENTS

    for folder in ("agents", "customer", "integration"):
        for path in (Path("casepilot") / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "emit"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    assert node.args[0].value in EVENTS, (str(path), node.args[0].value)


@pytest.mark.parametrize("method", ["invoke", "run", "ainvoke", "arun"])
def test_public_tool_entrypoints_disable_ambient_tracing(monkeypatch, method):
    import asyncio

    import langchain_core.tracers.langchain as tracing
    from langchain_core.callbacks import BaseCallbackHandler

    from casepilot.integration.tau2.backend import GuardedRetailTool

    captured = []

    class Tracer(BaseCallbackHandler):
        def __init__(self, *args, **kwargs):
            self.run_map = {}
            self.order_map = {}

        def on_tool_start(self, serialized, input_str, **kwargs):
            captured.append(input_str)

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(tracing, "LangChainTracer", Tracer)

    def echo(email: str) -> str:
        return email

    tool = GuardedRetailTool.from_function(echo, description="Read a record")
    result = getattr(tool, method)({"email": "private@example.com"})
    if method.startswith("a"):
        result = asyncio.run(result)
    assert result == "private@example.com"
    assert captured == []
