import json
import os


async def serve(kind):
    import anyio
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    from casepilot.mcp.bridge import request_from_environment

    server = Server(f"casepilot-retail-{kind}")

    @server.list_tools()
    async def list_tools():
        payload = await anyio.to_thread.run_sync(
            lambda: request_from_environment({"action": "list_tools", "kind": kind})
        )
        if "protocol_error" in payload:
            raise ValueError("Episode registry unavailable")
        return [Tool(**tool) for tool in payload["tools"]]

    @server.call_tool()
    async def call_tool(name, arguments):
        payload = await anyio.to_thread.run_sync(
            lambda: request_from_environment(
                {
                    "action": "call_tool",
                    "kind": kind,
                    "name": name,
                    "arguments": arguments,
                    "operation_id": os.environ["CASEPILOT_OPERATION_ID"],
                }
            )
        )
        if "protocol_error" in payload:
            raise ValueError("Episode tool request rejected")
        return [TextContent(type="text", text=json.dumps(payload))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
