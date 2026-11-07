"""Observable runtime contracts, independent of benchmark or provider libraries.

Model adapters translate these to LangChain messages; benchmark adapters translate them to tau2.
No simulator instructions, audio or evaluator payloads are accepted by the text runtime.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any]
    requestor: Literal["assistant", "user"] = "assistant"


class SystemMessage(BaseModel):
    role: Literal["system"] = "system"
    content: str | None = None


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str | None = None


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: dict[str, Any] | None = None
    cost: float | None = None


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    id: str
    content: str | None = None
    error: bool = False
    requestor: Literal["assistant", "user"] = "assistant"
    execution_status: str | None = None


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]


class MultiToolMessage(BaseModel):
    tool_messages: list[ToolMessage]


def normalize_message(message):
    """Deserialize only observable text, also accepting adapter-supplied Pydantic values."""
    if isinstance(message, MultiToolMessage):
        return message
    if not isinstance(message, dict):
        if getattr(message, "is_audio", False):
            raise ValueError("The runtime accepts text messages only")
        message = message.model_dump()
    if message.get("audio_content") or message.get("audio"):
        raise ValueError("The runtime accepts text messages only")
    if "tool_messages" in message:
        return MultiToolMessage(
            tool_messages=[normalize_message(m) for m in message["tool_messages"]]
        )
    if message.get("role") == "user" and message.get("tool_calls"):
        raise ValueError("User tool calls are not observable text input")
    return TypeAdapter(Message).validate_python(message)
