"""Project messages and tools cross into LangChain only at this boundary."""

import asyncio
import hashlib
import json
import secrets
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import DEFAULT_CONNECTION_LIMITS, DEFAULT_TIMEOUT

from casepilot.config import ProviderSettings, load_provider_settings
from casepilot.messages import AssistantMessage, ToolCall


def _messages(messages):
    converted = []
    for message in messages:
        role = message.role
        content = message.content or ""
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(
                AIMessage(
                    content=content,
                    tool_calls=[
                        {
                            "id": call.id,
                            "name": call.name,
                            "args": call.arguments,
                            "type": "tool_call",
                        }
                        for call in (message.tool_calls or [])
                    ],
                )
            )
        elif role == "tool":
            converted.append(ToolMessage(content=content, tool_call_id=message.id))
        else:
            raise ValueError(f"Unsupported message role: {role}")
    return converted


def _await_cleanup(coroutine):
    """The synchronous adapter never uses async transports on the caller's loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coroutine)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(asyncio.run, coroutine).result()


@dataclass(repr=False)
class _Client:
    chat: object
    sync: httpx.Client | None = None
    asynchronous: httpx.AsyncClient | None = None

    def close(self):
        errors = []
        for owned, root in (
            (self.sync, getattr(self.chat, "root_client", None)),
            (self.asynchronous, getattr(self.chat, "root_async_client", None)),
        ):
            if owned is None:
                continue
            try:
                if isinstance(owned, httpx.AsyncClient):
                    _await_cleanup(root.close() if root is not None else owned.aclose())
                else:
                    (root if root is not None else owned).close()
            except Exception as error:
                errors.append(error)
            finally:
                # SDK cleanup failures must not strand the transport we own.
                try:
                    if not owned.is_closed:
                        if isinstance(owned, httpx.AsyncClient):
                            _await_cleanup(owned.aclose())
                        else:
                            owned.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]


class OpenAIChatAdapter:
    """Bounded OpenAI client cache; calls and cleanup are serialized.

    Caller-injected SDK clients and HTTP transports retain caller ownership.
    """

    def __init__(self, settings: ProviderSettings | None = None, *, cache_size: int = 4):
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self._settings = settings.model_copy(deep=True) if settings is not None else None
        self._cache_size = cache_size
        self._clients = OrderedDict()
        self._key_salt = secrets.token_bytes(32)
        self._lock = RLock()
        self._closed = False

    def _client(self, settings, model, options):
        payload = json.dumps(
            [
                settings.api_key.get_secret_value(),
                settings.base_url.get_secret_value() if settings.base_url else None,
                model,
                options,
            ],
            sort_keys=True,
            default=repr,
        )
        key = hashlib.blake2b(payload.encode(), key=self._key_salt).hexdigest()
        if key in self._clients:
            self._clients.move_to_end(key)
            return self._clients[key].chat
        options = dict(options)
        owned_sync = owned_async = None
        transport_options = {
            "timeout": options.get("timeout", options.get("request_timeout", DEFAULT_TIMEOUT)),
            "limits": DEFAULT_CONNECTION_LIMITS,
        }
        if options.get("openai_proxy"):
            transport_options["proxy"] = options["openai_proxy"]
        # LangChain's defaults use globally cached transports: supply owned ones.
        if not any(options.get(k) is not None for k in ("http_client", "client", "root_client")):
            owned_sync = options["http_client"] = httpx.Client(**transport_options)
        if not any(
            options.get(k) is not None
            for k in ("http_async_client", "async_client", "root_async_client")
        ):
            owned_async = options["http_async_client"] = httpx.AsyncClient(**transport_options)
        if options.get("openai_proxy") and owned_sync is not None and owned_async is not None:
            options.pop("openai_proxy")
        try:
            chat = ChatOpenAI(
                model=model,
                api_key=settings.api_key,
                base_url=settings.base_url.get_secret_value() if settings.base_url else None,
                **options,
            )
        except Exception:
            _Client(None, owned_sync, owned_async).close()
            raise
        # A factory may substitute a caller-owned transport. Release our unused
        # allocations while preserving the replacement's ownership.
        if owned_sync is not None and chat.http_client is not owned_sync:
            owned_sync.close()
            owned_sync = None
        if owned_async is not None and chat.http_async_client is not owned_async:
            _await_cleanup(owned_async.aclose())
            owned_async = None
        self._clients[key] = _Client(chat, owned_sync, owned_async)
        if len(self._clients) > self._cache_size:
            _, evicted = self._clients.popitem(last=False)
            evicted.close()
        return chat

    def close(self):
        """Release owned sync/async clients once; further calls are rejected."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = list(self._clients.values())
            self._clients.clear()
            errors = []
            for entry in entries:
                try:
                    entry.close()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]

    def __call__(self, model, messages, tools, **kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAIChatAdapter is closed")
            return self._call(model, messages, tools, **kwargs)

    def _call(self, model, messages, tools, **kwargs):
        settings = self._settings or load_provider_settings()
        options = dict(kwargs)
        options.pop("call_name", None)
        options.pop("allowed_openai_params", None)
        if "num_retries" in options:
            options["max_retries"] = options.pop("num_retries")
        if "max_completion_tokens" in options:
            options["max_tokens"] = options.pop("max_completion_tokens")
        binding = {
            key: options.pop(key)
            for key in ("response_format", "parallel_tool_calls", "tool_choice")
            if key in options
        }
        model = model.removeprefix("openai/")
        options.setdefault("use_responses_api", False)
        if model.startswith("responses/"):
            model = model.removeprefix("responses/")
            options["use_responses_api"] = True
            # The Responses SDK does not accept Chat Completions' seed parameter.
            options.pop("seed", None)
        if model.startswith("gpt-6"):
            options.setdefault("reasoning_effort", "none")
        # Ambient LangSmith/LangChain variables must never upload raw model data.
        with tracing_context(enabled=False):
            chat = self._client(settings, model, options)
            if tools:
                from casepilot.tools.definitions import tool_schema

                declarations = [tool_schema(tool) for tool in tools]
                chat = chat.bind_tools(declarations, **binding)
            elif binding:
                binding.pop("parallel_tool_calls", None)
                binding.pop("tool_choice", None)
                if binding:
                    chat = chat.bind(**binding)
            reply = chat.invoke(_messages(messages))
        if reply.invalid_tool_calls:
            raise ValueError("Provider returned invalid tool arguments")
        content = reply.content
        if isinstance(content, list):
            content = "".join(
                block if isinstance(block, str) else block.get("text", "") for block in content
            )
        usage = None
        if reply.usage_metadata is not None:
            usage = dict(reply.usage_metadata)
            usage["prompt_tokens"] = usage["input_tokens"]
            usage["completion_tokens"] = usage["output_tokens"]
        return AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=[
                ToolCall(id=call["id"], name=call["name"], arguments=call["args"])
                for call in reply.tool_calls
            ],
            usage=usage,
        )
