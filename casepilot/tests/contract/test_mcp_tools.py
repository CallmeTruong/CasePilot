def test_read_action_schemas_partition_pinned_registry(retail_environment):
    from casepilot.mcp.tool_registry import describe_tools

    read = {t["name"]: t for t in describe_tools(retail_environment, "read")}
    action = {t["name"]: t for t in describe_tools(retail_environment, "action")}
    expected = {t.name for t in retail_environment.get_tools()}
    assert not read.keys() & action.keys()
    assert read.keys() | action.keys() == expected
    assert "find_user_id_by_email" in read
    assert "exchange_delivered_order_items" in action
    assert "transfer_to_human_agents" in action
    for tool in retail_environment.get_tools():
        record = (read | action)[tool.name]
        assert record["inputSchema"] == tool.openai_schema["function"]["parameters"]
