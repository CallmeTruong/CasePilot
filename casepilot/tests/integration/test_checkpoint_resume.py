import pytest

from casepilot.agents.state import CaseState


def test_checkpoint_scope_backend_and_pending_receipt_guard(tmp_path, retail_environment):
    from tau2.data_model.message import AssistantMessage, ToolCall

    from casepilot.memory.checkpoint import CheckpointManager
    from casepilot.reliability.operations import OperationJournal

    env = retail_environment
    manager = CheckpointManager(
        tmp_path / "checkpoints.sqlite", benchmark_revision="revision", config_hash="config"
    )
    journal = OperationJournal(tmp_path / "operations.sqlite")
    customer = next(iter(env.tools.db.users))
    args = {
        "user_id": customer,
        "address1": "Checkpoint fixture street",
        "address2": "",
        "city": "Test",
        "state": "CA",
        "country": "USA",
        "zip": "90001",
    }
    call = ToolCall(id="op", name="modify_user_address", arguments=args)
    state = CaseState(
        episode_id="episode",
        customer_id=customer,
        messages=[AssistantMessage(role="assistant", tool_calls=[call])],
        pending_tool_calls={"op": {"name": call.name, "arguments": args}},
    )
    journal.start("episode", "op", call.name, args)
    result = env.get_response(call)
    assert not result.error
    journal.finish(
        "episode",
        "op",
        "committed",
        proof="receipt",
        result=result.model_dump(mode="json"),
        backend_hash=env.get_db_hash(),
    )
    manager.save(state, env.get_db_hash())
    restored, receipts = manager.restore("episode", backend_hash=env.get_db_hash(), journal=journal)
    from casepilot.messages import normalize_message

    assert receipts == [normalize_message(result)]
    assert restored.messages[0].tool_calls[0].id == "op"
    assert journal.get("episode", "op")["attempt"] == 1  # receipt, no dispatch/replay
    with pytest.raises(ValueError, match="Backend state"):
        manager.restore("episode", backend_hash="fresh-reset-db", journal=journal)
    with pytest.raises(ValueError, match="revision/config"):
        manager.restore("episode", backend_hash=env.get_db_hash(), config_hash="other")
    with pytest.raises(ValueError, match="No checkpoint"):
        manager.restore("other-episode", backend_hash=env.get_db_hash())
    manager.close()
    journal.close()


def test_langgraph_checkpoint_roundtrip_keeps_observable_messages(tmp_path, retail_environment):
    from tau2.data_model.message import AssistantMessage, UserMessage

    from casepilot.config import RunConfig
    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment
    from casepilot.memory.checkpoint import CheckpointManager

    config = RunConfig(model="openai/gpt-6-luna")
    manager = CheckpointManager(
        tmp_path / "graph.sqlite", benchmark_revision="revision", config_hash=config.fingerprint()
    )
    agent = CasePilotAgent(
        retail_environment.get_tools(),
        retail_environment.get_policy(),
        config,
        checkpoint_manager=manager,
        generate=lambda **_: AssistantMessage(role="assistant", content="How can I help?"),
    )
    wrapper = GuardedEnvironment(retail_environment, agent, journal_path=tmp_path / "ops.sqlite")
    try:
        reply, state = agent.generate_next_message(
            UserMessage(role="user", content="Hello"), agent.get_init_state()
        )
        snapshot = agent.engine.graph.get_state({"configurable": {"thread_id": agent.episode_id}})
        assert snapshot.values["case"].episode_id == agent.episode_id
        restored, receipts = manager.restore(agent.episode_id, backend_hash=wrapper.get_db_hash())
        assert receipts == []
        assert restored.messages[0].role == "user"
        assert restored.messages[-1].content == reply.content
        assert not restored.pending_tool_calls
    finally:
        wrapper.close()
