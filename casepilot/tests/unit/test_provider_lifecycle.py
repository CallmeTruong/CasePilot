"""Adapter client reuse and ownership, without provider network access."""

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

from casepilot.config import ProviderSettings
from casepilot.llm import openai_chat_adapter as provider
from casepilot.messages import UserMessage


@pytest.fixture
def clients(monkeypatch):
    created = []
    actual = provider.ChatOpenAI

    def build(**kwargs):
        chat = actual(**kwargs)
        created.append(chat)
        return chat

    monkeypatch.setattr(provider, "ChatOpenAI", build)
    monkeypatch.setattr(actual, "invoke", lambda *a, **kw: AIMessage(content="ok"))
    return created


def call(adapter, **kwargs):
    return adapter("gpt-6", [UserMessage(content="private prompt")], [], **kwargs)


def test_client_reuse_and_binding_do_not_cache_prompts(clients):
    adapter = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("secret-key")))
    call(adapter, timeout=10, num_retries=0)
    call(adapter, timeout=10, num_retries=0, response_format={"type": "json_object"})
    adapter(
        "gpt-6",
        [UserMessage(content="different private prompt")],
        [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
            }
        ],
        timeout=10,
        num_retries=0,
        tool_choice="auto",
    )
    assert len(clients) == 1
    assert "secret-key" not in repr(adapter.__dict__)
    assert "private prompt" not in repr(adapter.__dict__)
    adapter.close()


def test_configuration_changes_and_lru_eviction_release_both_clients(clients):
    adapter = provider.OpenAIChatAdapter(
        ProviderSettings(api_key=SecretStr("secret-key")), cache_size=2
    )
    call(adapter, timeout=10)
    call(adapter, timeout=20)
    call(adapter, timeout=10)
    call(adapter, timeout=30)
    assert len(clients) == 3
    assert clients[1].root_client.is_closed()
    assert clients[1].root_async_client.is_closed()
    assert not clients[0].root_client.is_closed()
    call(adapter, timeout=30, reasoning_effort="low")
    assert len(clients) == 4
    adapter.close()
    assert all(c.root_client.is_closed() and c.root_async_client.is_closed() for c in clients)


def test_close_is_idempotent_and_works_inside_running_loop(clients):
    adapter = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("secret-key")))
    call(adapter)

    async def close():
        adapter.close()
        adapter.close()

    asyncio.run(close())
    assert clients[0].root_client.is_closed() and clients[0].root_async_client.is_closed()
    with pytest.raises(RuntimeError, match="closed"):
        call(adapter)


def test_injected_transports_remain_open(clients):
    sync = httpx.Client()
    asynchronous = httpx.AsyncClient()
    adapter = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("secret-key")))
    call(adapter, http_client=sync, http_async_client=asynchronous)
    adapter.close()
    assert not sync.is_closed and not asynchronous.is_closed
    sync.close()
    asyncio.run(asynchronous.aclose())


def test_explicit_settings_are_snapshotted_and_canonical_import_is_shared(clients):
    from casepilot.llm import openai_chat_adapter

    assert provider is openai_chat_adapter
    settings = ProviderSettings(api_key=SecretStr("original-key"))
    adapter = openai_chat_adapter.OpenAIChatAdapter(settings)
    settings.api_key = SecretStr("mutated-key")
    call(adapter)
    assert clients[0].openai_api_key.get_secret_value() == "original-key"
    adapter.close()


@pytest.mark.parametrize(
    "options",
    [
        {"num_retries": 0},
        {"seed": 42},
        {"reasoning_effort": "low"},
        {"use_responses_api": True},
        {"reasoning": {"effort": "low", "summary": "auto"}},
    ],
)
def test_constructor_configuration_is_part_of_cache_identity(clients, options):
    adapter = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("secret-key")))
    call(adapter)
    call(adapter, **options)
    call(adapter, **options)
    assert len(clients) == 2
    adapter.close()


def test_environment_settings_changes_get_new_client_without_exposing_values(clients, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "first-private-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://private-first.invalid/v1")
    adapter = provider.OpenAIChatAdapter()
    call(adapter)
    monkeypatch.setenv("OPENAI_API_KEY", "second-private-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://private-second.invalid/v1")
    call(adapter)
    call(adapter)
    assert len(clients) == 2
    assert clients[1].root_client.api_key == "second-private-key"
    assert str(clients[1].root_client.base_url) == "https://private-second.invalid/v1/"
    assert "private" not in repr(adapter.__dict__)
    adapter.close()


def test_closing_one_adapter_does_not_close_another_adapters_clients(clients):
    settings = ProviderSettings(api_key=SecretStr("secret-key"))
    first, second = provider.OpenAIChatAdapter(settings), provider.OpenAIChatAdapter(settings)
    call(first)
    call(second)
    first.close()
    assert clients[0].root_client.is_closed()
    assert not clients[1].root_client.is_closed()
    assert not clients[1].root_async_client.is_closed()
    assert call(second).content == "ok"
    second.close()


def test_close_releases_all_transports_even_when_one_sdk_close_fails(clients, monkeypatch):
    adapter = provider.OpenAIChatAdapter(ProviderSettings(api_key=SecretStr("secret-key")))
    call(adapter, timeout=10)
    call(adapter, timeout=20)

    def broken_close():
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(clients[0].root_client, "close", broken_close)
    with pytest.raises(RuntimeError, match="cleanup failure"):
        adapter.close()
    assert clients[0].http_client.is_closed
    assert clients[0].http_async_client.is_closed
    assert clients[1].http_client.is_closed
    assert clients[1].http_async_client.is_closed
    adapter.close()
