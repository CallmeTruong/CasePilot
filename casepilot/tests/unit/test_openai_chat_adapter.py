"""The provider boundary sends native OpenAI requests without contacting a provider."""

import json

import httpx
import pytest
from langchain_core.tools import StructuredTool
from pydantic import SecretStr

from casepilot.config import ProviderSettings


def test_native_tool_request_and_usage(monkeypatch):
    from casepilot.llm import openai_chat_adapter as provider
    from casepilot.messages import (
        AssistantMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_next",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"order_id":"o2"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29},
            },
        )

    actual = provider.ChatOpenAI
    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(
        provider, "ChatOpenAI", lambda **kw: actual(**{**kw, "http_client": client})
    )
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    # Ambient tracing must never create a client or send a trace.
    import langsmith.run_trees

    monkeypatch.setattr(
        langsmith.run_trees, "get_cached_client", lambda **kw: pytest.fail("implicit tracing")
    )

    def lookup(order_id: str) -> str:
        """Look up an order."""
        return order_id

    reply = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("private-key")))(
        model="openai/gpt-6",
        messages=[
            SystemMessage(role="system", content="System"),
            UserMessage(role="user", content="Find it"),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="call_old", name="lookup", arguments={"order_id": "o1"})],
            ),
            ToolMessage(role="tool", content="done", id="call_old"),
        ],
        tools=[StructuredTool.from_function(lookup)],
        temperature=0,
        seed=42,
        timeout=10,
        num_retries=0,
        max_completion_tokens=123,
        reasoning_effort="none",
        parallel_tool_calls=False,
        allowed_openai_params=["reasoning_effort"],
        call_name="planner",
    )
    payload = requests[0]
    assert payload["model"] == "gpt-6"
    assert payload["tools"][0]["function"]["name"] == "lookup"
    assert payload["messages"][2]["tool_calls"][0]["id"] == "call_old"
    assert payload["messages"][3] == {"content": "done", "role": "tool", "tool_call_id": "call_old"}
    assert payload["max_completion_tokens"] == 123
    assert payload["reasoning_effort"] == "none"
    assert payload["parallel_tool_calls"] is False
    assert payload["temperature"] == 0 and payload["seed"] == 42
    assert reply.tool_calls[0].arguments == {"order_id": "o2"}
    assert reply.usage["prompt_tokens"] == 21
    assert reply.usage["completion_tokens"] == 8


def test_planner_without_tools_and_honest_provider_errors(monkeypatch):
    from openai import BadRequestError

    from casepilot.llm import openai_chat_adapter as provider
    from casepilot.messages import UserMessage

    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "bad model",
                    "type": "invalid_request_error",
                    "code": "bad_model",
                }
            },
        )

    actual = provider.ChatOpenAI
    monkeypatch.setattr(
        provider,
        "ChatOpenAI",
        lambda **kw: actual(
            **{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(respond))}
        ),
    )
    with pytest.raises(BadRequestError):
        provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("private-key")))(
            model="gpt-6",
            messages=[UserMessage(role="user", content="Plan")],
            tools=[],
            response_format={"type": "json_object"},
            num_retries=0,
        )
    assert "tools" not in payloads[0]
    assert payloads[0]["response_format"] == {"type": "json_object"}


def test_explicit_responses_model_uses_responses_endpoint(monkeypatch):
    from casepilot.llm import openai_chat_adapter as provider
    from casepilot.messages import UserMessage

    requests = []

    def respond(request):
        requests.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "instructions": None,
                "model": "gpt-6",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "temperature": 0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1,
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    actual = provider.ChatOpenAI
    monkeypatch.setattr(
        provider,
        "ChatOpenAI",
        lambda **kw: actual(
            **{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(respond))}
        ),
    )
    reply = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("private-key")))(
        model="responses/gpt-6",
        messages=[UserMessage(content="Hello")],
        tools=[],
        seed=42,
        num_retries=0,
        max_completion_tokens=100,
    )
    assert requests[0][0] == "/v1/responses"
    assert requests[0][1]["model"] == "gpt-6"
    assert requests[0][1]["max_output_tokens"] == 100
    assert "seed" not in requests[0][1]
    assert reply.content == "Hello" and reply.usage["completion_tokens"] == 2


def test_malformed_provider_tool_arguments_are_not_silently_dropped(monkeypatch):
    from casepilot.llm import openai_chat_adapter as provider
    from casepilot.messages import UserMessage

    def respond(request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-bad",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-6",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "not json"},
                                }
                            ],
                        },
                    }
                ],
            },
        )

    actual = provider.ChatOpenAI
    monkeypatch.setattr(
        provider,
        "ChatOpenAI",
        lambda **kw: actual(
            **{**kw, "http_client": httpx.Client(transport=httpx.MockTransport(respond))}
        ),
    )
    with pytest.raises(ValueError, match="invalid tool"):
        provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("private-key")))(
            model="gpt-6",
            messages=[UserMessage(content="Look up")],
            tools=[],
            num_retries=0,
        )
