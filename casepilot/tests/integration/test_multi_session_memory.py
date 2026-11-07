from datetime import UTC, datetime

from casepilot.config import RunConfig
from casepilot.memory.episodic import CaseHistory, HistoryStore


def test_history_is_hidden_until_identity_and_disabled_in_m0(tmp_path, retail_environment):
    from tau2.data_model.message import UserMessage

    from casepilot.integration.tau2.benchmark_agent import CasePilotAgent

    store = HistoryStore(tmp_path / "memory.sqlite")
    record = CaseHistory(
        history_id="history",
        memory_scenario_id="synthetic-scope",
        customer_id="verified",
        case_id="PREVIOUS_CASE_NONCE",
        source_episode_id="old-episode",
        session_id="old-session",
        intent="lookup",
        result="profile_verified",
        last_action="get_user_details",
        verified=True,
        verified_at=datetime.now(UTC),
        source_operation_ids=["source-operation"],
    )
    store.put(record)
    env = retail_environment
    for mode in ("M0", "M1"):
        config = RunConfig(track="custom", model="openai/gpt-6-luna", memory=mode)
        agent = CasePilotAgent(
            env.get_tools(),
            env.get_policy(),
            config,
            memory_store=store,
            memory_scope="synthetic-scope",
            generate=lambda **_: None,
        )
        state = agent.get_init_state()
        state.messages = [UserMessage(role="user", content="What happened last session?")]
        assert "PREVIOUS_CASE_NONCE" not in agent.engine.prompt(state)[0].content
        state.customer_id = "verified"
        system_prompt = agent.engine.prompt(state)[0].content
        assert ("PREVIOUS_CASE_NONCE" in system_prompt) is (mode == "M1")
        if mode == "M1":
            assert "may be stale" in system_prompt
    store.close()
