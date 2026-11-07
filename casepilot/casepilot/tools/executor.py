"""One private framework catalog, invoked only after policy and journal admission."""

from collections.abc import Callable, Mapping, Sequence

from langchain_core.tools import StructuredTool
from langsmith import tracing_context

from casepilot.tools.definitions import ToolDefinition


class ToolExecutor:
    def __init__(
        self,
        definitions: Sequence[ToolDefinition],
        handlers: Mapping[str, Callable],
    ):
        self.definitions = tuple(definitions)
        names = [item.name for item in self.definitions]
        if len(set(names)) != len(names) or set(names) != set(handlers):
            raise ValueError("Tool definitions and handlers must match uniquely")
        self._tools = {
            item.name: StructuredTool.from_function(
                func=handlers[item.name],
                name=item.name,
                description=item.description or item.name,
                args_schema=item.args_schema,
                infer_schema=False,
            )
            for item in self.definitions
        }

    def invoke(self, name: str, arguments: dict) -> str:
        with tracing_context(enabled=False):
            return self._tools[name].invoke(arguments)
