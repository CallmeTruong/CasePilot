"""Benchmark-only adapter. Core Agent never imports the tau2 protocol."""

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import AssistantMessage

from casepilot.agents.agent import Agent
from casepilot.agents.state import CaseState


class CasePilotAgent(Agent, HalfDuplexAgent[CaseState]):
    """Keep upstream Orchestrator compatibility at one explicit boundary."""

    def __init__(self, tools, *args, **kwargs):
        from langchain_core.tools import BaseTool

        from casepilot.integration.tau2.tools import describe_tools
        from casepilot.tools.definitions import ToolDefinition

        declarations = [
            tool
            if isinstance(tool, (ToolDefinition, BaseTool, dict))
            else describe_tools([tool])[0]
            for tool in tools
        ]
        super().__init__(declarations, *args, **kwargs)

    def generate_next_message(self, message, state):
        reply, state = super().generate_next_message(message, state)
        payload = reply.model_dump(exclude_none=True)
        # tau2 tests presence rather than truthiness when routing tool messages.
        # LangChain's normal empty list must become tau2's absent-tool sentinel.
        if not payload.get("tool_calls"):
            payload["tool_calls"] = None
        return AssistantMessage.model_validate(payload), state
