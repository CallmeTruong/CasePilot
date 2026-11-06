from casepilot.agents.verifier import MUTATION_TOOLS

ACTION_TOOLS = MUTATION_TOOLS | {"transfer_to_human_agents"}


def describe_tools(environment, kind):
    if kind not in {"read", "action"}:
        raise ValueError("MCP tool group must be read or action")
    result = []
    for tool in environment.get_tools():
        mutation = tool.name in ACTION_TOOLS
        if (kind == "action") != mutation:
            continue
        schema = tool.openai_schema["function"]
        result.append(
            {
                "name": tool.name,
                "description": schema["description"],
                "inputSchema": schema["parameters"],
            }
        )
    return result
