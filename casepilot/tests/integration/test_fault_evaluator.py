from types import SimpleNamespace


def test_fault_grading_uses_actual_state_and_never_replays_failed_call(retail_environment):
    from tau2.domains.retail.environment import get_tasks

    from casepilot.evaluation.fault_evaluator import evaluate_fault_simulation

    task = next(t for t in get_tasks(task_split_name=None) if t.id == "0")
    simulation = SimpleNamespace(messages=[], reward_info=None)
    orchestrator = SimpleNamespace(
        run=lambda: simulation, task=task, environment=retail_environment
    )
    initial = task.initial_state
    retail_environment.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=initial.message_history if initial else [],
    )
    before = retail_environment.get_db_hash()
    result = evaluate_fault_simulation(orchestrator)
    assert result.reward_info.reward == 0
    assert result.reward_info.info["trajectory_replay"] is False
    assert retail_environment.get_db_hash() == before
    for action in task.evaluation_criteria.actions:
        retail_environment.make_tool_call(
            tool_name=action.name, requestor=action.requestor, **action.arguments
        )
    assert evaluate_fault_simulation(orchestrator).reward_info.reward == 1


def test_invalid_reference_read_matches_upstream_gold_replay_semantics(
    retail_environment, monkeypatch
):
    from tau2.data_model.simulation import RewardInfo
    from tau2.domains.retail.environment import get_tasks
    from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

    from casepilot.evaluation.fault_evaluator import evaluate_fault_simulation

    monkeypatch.setattr(
        NLAssertionsEvaluator, "calculate_reward", lambda **_kwargs: RewardInfo(reward=1.0)
    )
    task = next(t for t in get_tasks(task_split_name=None) if t.id == "2")
    simulation = SimpleNamespace(messages=[], reward_info=None)
    orchestrator = SimpleNamespace(
        run=lambda: simulation, task=task, environment=retail_environment
    )
    result = evaluate_fault_simulation(orchestrator)
    errors = result.reward_info.info["gold_action_errors"]
    assert any(
        e["tool_name"] == "get_product_details" and e["error_type"] == "ValueError" for e in errors
    )
    assert result.reward_info.reward in {0.0, 1.0}
