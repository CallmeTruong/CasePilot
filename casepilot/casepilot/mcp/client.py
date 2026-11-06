import asyncio
import json
import os
import sys


class MCPToolClient:
    def __init__(self, bridge, *, timeout_seconds=15):
        self.bridge, self.timeout_seconds = bridge, timeout_seconds

    async def call(self, kind, name, arguments, operation_id):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        env = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE")
            if key in os.environ
        }
        env.update(
            CASEPILOT_BRIDGE_PORT=str(self.bridge.address[1]),
            CASEPILOT_BRIDGE_KEY=self.bridge.authkey.hex(),
            CASEPILOT_EPISODE_ID=self.bridge.episode_id,
            CASEPILOT_OPERATION_ID=operation_id,
        )
        params = StdioServerParameters(
            command=sys.executable, args=["-m", f"casepilot.mcp.{kind}_server"], env=env
        )
        async with asyncio.timeout(max(60, self.timeout_seconds)):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    async with asyncio.timeout(self.timeout_seconds):
                        result = await session.call_tool(name, arguments)
                    if result.isError:
                        raise RuntimeError("MCP call rejected")
                    return json.loads(result.content[0].text)

    def call_sync(self, kind, name, arguments, operation_id):
        return asyncio.run(self.call(kind, name, arguments, operation_id))
