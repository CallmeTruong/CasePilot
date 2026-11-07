"""Immutable tool declarations without executable capabilities or backend assumptions."""

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    args_schema: type[BaseModel] | dict = field(repr=False, compare=False)
    _schema_json: str = field(init=False, repr=False)

    def __post_init__(self):
        schema = (
            self.args_schema.model_json_schema()
            if isinstance(self.args_schema, type)
            else self.args_schema
        )
        object.__setattr__(self, "_schema_json", json.dumps(schema, sort_keys=True))

    @property
    def input_schema(self):
        return json.loads(self._schema_json)

    @property
    def openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


def describe_tool(tool: ToolDefinition | BaseTool | dict) -> ToolDefinition:
    if isinstance(tool, ToolDefinition):
        return tool
    if isinstance(tool, BaseTool):
        schema = convert_to_openai_tool(tool)["function"]
    elif isinstance(tool, dict):
        schema = tool.get("function", tool)
    else:
        raise TypeError("Use ToolDefinition, a LangChain tool, or a provider declaration")
    return ToolDefinition(schema["name"], schema.get("description", ""), schema["parameters"])


def tool_schema(tool: Any) -> dict:
    return describe_tool(tool).openai_schema
