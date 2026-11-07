def test_an_episode_write_lock_does_not_block_another_episode(
    retail_environment, tmp_path, monkeypatch
):
    from tau2.domains.retail.environment import get_tasks

    from casepilot.config import RunConfig
    from casepilot.integration.simulation_runner import make_casepilot_orchestrator

    monkeypatch.chdir(tmp_path)
    task = next(t for t in get_tasks(task_split_name=None) if t.id == "0")
    config = RunConfig(model="openai/gpt-6-luna", user_model="openai/gpt-6-luna")
    episodes = [
        make_casepilot_orchestrator(
            config, task, seed=42, simulation_id=name, benchmark_revision="test-revision"
        )
        for name in ("first-episode", "second-episode")
    ]
    try:
        first, second = [episode.agent.checkpoint_manager for episode in episodes]
        first.connection.execute("BEGIN IMMEDIATE")
        state = episodes[1].agent.get_init_state()
        backend_hash = episodes[1].environment.get_db_hash()
        second.save(state, backend_hash)
        restored, receipts = second.restore(state.episode_id, backend_hash=backend_hash)
        assert restored.episode_id == "second-episode" and not receipts
    finally:
        episodes[0].agent.checkpoint_manager.connection.rollback()
        for episode in episodes:
            episode.environment.close()
