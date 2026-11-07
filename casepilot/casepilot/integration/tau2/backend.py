"""Retail loading and tau2 message conversion, isolated from the core runtime."""

from pathlib import Path
from types import MappingProxyType
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from langsmith import tracing_context


class GuardedRetailTool(StructuredTool):
    """Public framework invocation keeps the same privacy and authorization boundary."""

    def invoke(self, input, config=None, **kwargs):
        with tracing_context(enabled=False):
            return super().invoke(input, config, **kwargs)

    def run(self, *args, **kwargs):
        with tracing_context(enabled=False):
            return super().run(*args, **kwargs)

    async def arun(self, *args, **kwargs):
        with tracing_context(enabled=False):
            return await super().arun(*args, **kwargs)


class RetailBackend:
    """Project-facing facade over the shared guarded native/MCP dispatcher."""

    def __init__(self, guarded_environment):
        self._environment = guarded_environment
        self._public_tools = None

    @property
    def agent(self):
        return self._environment.agent

    @property
    def tool_definitions(self):
        return self._environment.tool_definitions

    @property
    def dispatch_count(self):
        return self._environment.dispatch_count

    @property
    def mutation_commits(self):
        return MappingProxyType(self._environment.mutation_commits)

    @property
    def transport_errors(self):
        return self._environment.transport_errors

    @property
    def recovery_steps(self):
        return self._environment.recovery_steps

    def snapshot(self):
        return self._environment.tools.db.model_dump(mode="json")

    def operation_records(self):
        return self._environment.journal.records(self.agent.episode_id)

    def close(self):
        self._environment.close()

    def as_tools(self):
        from casepilot.messages import ToolCall

        if self._public_tools is not None:
            return self._public_tools

        def bind(name):
            def execute(**arguments):
                response = self.execute(ToolCall(id=uuid4().hex, name=name, arguments=arguments))
                if response.error:
                    raise ToolException(response.content or "Tool request rejected")
                agent = self._environment.agent
                agent.engine.gate.observe_result(
                    name, arguments, response.content, response.error, agent.current_state
                )
                return response.content

            return execute

        self._public_tools = tuple(
            GuardedRetailTool.from_function(
                func=bind(tool.name),
                name=tool.name,
                description=tool.description,
                args_schema=tool.args_schema,
                infer_schema=False,
            )
            for tool in self.tool_definitions
        )
        return self._public_tools

    def execute(self, message):
        from tau2.data_model.message import ToolCall as RetailToolCall

        from casepilot.messages import ToolMessage

        request = RetailToolCall(
            id=message.id,
            name=message.name,
            arguments=message.arguments,
            requestor=getattr(message, "requestor", "assistant"),
        )
        response = self._environment.get_response(request)
        return ToolMessage(
            id=response.id,
            content=response.content,
            error=response.error,
            requestor=response.requestor,
            execution_status=getattr(response, "execution_status", None),
        )


def load_retail_environment(manifest_path=None):
    """Verify pinned business sources before loading an isolated Retail database."""
    from casepilot.integration.benchmark import load_manifest, prepare_runtime

    path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path(__file__).resolve().parents[3] / "configs/benchmark-manifest.json"
    )
    prepare_runtime(load_manifest(path), path)
    from tau2.domains.retail.environment import get_environment
    from tau2.utils.llm_utils import set_llm_log_dir

    set_llm_log_dir(None)
    return get_environment()
