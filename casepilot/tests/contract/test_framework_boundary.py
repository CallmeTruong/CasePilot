import ast
from pathlib import Path

import pytest


def test_core_layers_do_not_import_tau2():
    violations = []
    for folder in ("agents", "customer", "memory"):
        for path in (Path("casepilot") / folder).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tau2"):
                    violations.append(str(path))
                if isinstance(node, ast.Import):
                    violations.extend(str(path) for a in node.names if a.name.startswith("tau2"))
    assert not violations, violations


def test_core_message_roundtrip_rejects_audio_and_preserves_tool_correlation():
    from casepilot.messages import AssistantMessage, ToolCall, normalize_message

    message = AssistantMessage(
        role="assistant", tool_calls=[ToolCall(id="call-1", name="read", arguments={"id": "A"})]
    )
    restored = normalize_message(message.model_dump())
    assert restored.tool_calls[0].id == "call-1"
    with pytest.raises(ValueError):
        normalize_message({"role": "user", "content": "hello", "audio_content": "encoded"})


def test_core_agent_runs_with_langchain_tools_when_tau2_imports_are_forbidden(monkeypatch):
    import builtins

    import langsmith.run_trees
    from langchain_core.tools import StructuredTool

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setattr(
        langsmith.run_trees,
        "get_cached_client",
        lambda **kwargs: pytest.fail("implicit graph tracing"),
    )
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tau2" or name.startswith("tau2."):
            raise AssertionError("Core must run independently of the benchmark package")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    from casepilot.agents.agent import Agent
    from casepilot.config import RunConfig
    from casepilot.messages import AssistantMessage, UserMessage

    def calculate(expression: str) -> str:
        """Describe a safe arithmetic request."""
        return expression

    tool = StructuredTool.from_function(calculate)
    agent = Agent(
        [tool],
        "Retail policy",
        RunConfig(model="openai/test"),
        generate=lambda **kwargs: AssistantMessage(content="Please provide your order email."),
    )
    reply, state = agent.generate_next_message(
        UserMessage(content="Help with my order"), agent.get_init_state()
    )
    assert reply.content == "Please provide your order email."
    assert state.llm_calls == 1


def test_initial_history_rejects_user_tool_results():
    from casepilot.agents.agent import Agent
    from casepilot.config import RunConfig
    from casepilot.messages import ToolMessage

    agent = Agent([], "policy", RunConfig(model="openai/test"))
    with pytest.raises(ValueError, match="Non-observable"):
        agent.get_init_state([ToolMessage(id="untrusted", requestor="user", content="private")])


def test_benchmark_adapter_does_not_mark_empty_tool_list_as_a_tool_call():
    from casepilot.config import RunConfig
    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent
    from casepilot.messages import AssistantMessage, UserMessage

    agent = CasePilotAgent(
        [],
        "policy",
        RunConfig(model="openai/test"),
        generate=lambda **kwargs: AssistantMessage(
            content="Please provide your email.", tool_calls=[]
        ),
    )
    reply, _ = agent.generate_next_message(UserMessage(content="Help"), agent.get_init_state())
    assert reply.content == "Please provide your email."
    assert not reply.is_tool_call()
