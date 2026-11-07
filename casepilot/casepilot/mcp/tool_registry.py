from casepilot.agents.verifier import MUTATION_TOOLS
from casepilot.tools.definitions import tool_schema

ACTION_TOOLS = MUTATION_TOOLS | {"transfer_to_human_agents"}


def describe_tools(environment, kind):
    if kind not in {"read", "action"}:
        raise ValueError("MCP tool group must be read or action")
    result = []
    declarations = getattr(environment, "tool_definitions", None)
    if declarations is None:
        from casepilot.integration.tau2.tools import describe_tools as describe_backend

        declarations = describe_backend(environment.get_tools())
    for tool in declarations:
        mutation = tool.name in ACTION_TOOLS
        if (kind == "action") != mutation:
            continue
        schema = tool_schema(tool)["function"]
        result.append(
            {
                "name": tool.name,
                "description": schema["description"],
                "inputSchema": schema["parameters"],
            }
        )
    return result
