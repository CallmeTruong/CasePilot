import asyncio

from casepilot.mcp.server import serve

if __name__ == "__main__":
    asyncio.run(serve("action"))
