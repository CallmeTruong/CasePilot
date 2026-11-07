"""Pinned Retail schema extraction and result encoding; no public raw tool builder."""

import json
from datetime import date, datetime

from pydantic import BaseModel


def _json_value(value):
    """Preserve the pinned Retail result contract, including string scalars."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise ValueError(f"Unsupported tool result type: {type(value).__name__}")


def describe_tools(declarations):
    from casepilot.tools.definitions import ToolDefinition

    return tuple(
        ToolDefinition(
            tool.name, tool.openai_schema["function"].get("description", tool.name), tool.params
        )
        for tool in declarations
    )


def create_executor(environment, definitions):
    from casepilot.tools.executor import ToolExecutor

    def bind(name):
        def execute(**arguments):
            result = environment.tools.use_tool(name, **arguments)
            return (
                result if isinstance(result, str) else json.dumps(_json_value(result), default=str)
            )

        return execute

    return ToolExecutor(definitions, {tool.name: bind(tool.name) for tool in definitions})
