import ast
from pathlib import Path

RETIRED = {
    "casepilot.integration.tool_adapter",
    "casepilot.integration.retail_backend",
    "casepilot.integration.tau2_agent",
    "casepilot.llm.langchain",
    "casepilot.observability.tracing",
    "casepilot.observability.langsmith",
    "casepilot.tools.catalog",
    "casepilot.mcp.fault_proxy",
    "casepilot.evaluation.runner",
    "casepilot.evaluation.tool_metrics",
    "casepilot.customer.workspace",
}


def test_retired_modules_are_removed_and_have_no_python_consumers():
    for module in RETIRED:
        assert not Path(*module.split(".")).with_suffix(".py").exists(), module
    for folder in ("casepilot", "dashboard", "scripts", "tests"):
        for path in Path(folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom):
                    assert node.module not in RETIRED, (str(path), node.module)
                    for alias in node.names:
                        assert f"{node.module}.{alias.name}" not in RETIRED, str(path)
                elif isinstance(node, ast.Import):
                    assert not RETIRED.intersection(a.name for a in node.names), str(path)


def test_application_has_no_raw_compatibility_tool_builder():
    from casepilot.integration.tau2 import tools

    assert not hasattr(tools, "build_tools")
    assert not hasattr(tools, "tool_schema")
