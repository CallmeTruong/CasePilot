"""Custom fault-track grading against live backend state, without replaying failed calls.

The ordinary upstream evaluator remains unchanged for all no-fault experiments.
"""


def evaluate_fault_simulation(orchestrator):
    from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
    from tau2.data_model.tasks import RewardType
    from tau2.domains.retail.environment import get_environment
    from tau2.evaluator.evaluator_action import ActionEvaluator
    from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator

    simulation = orchestrator.run()
    simulation.policy = orchestrator.environment.get_policy()
    task = orchestrator.task
    criteria = task.evaluation_criteria
    if criteria is None:
        raise ValueError("Fault experiment requires evaluator criteria")
    gold = get_environment()
    initial = task.initial_state
    gold.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=(initial.message_history or []) if initial else [],
    )
    gold_action_errors = []
    for action in criteria.actions or []:
        try:
            gold.make_tool_call(
                tool_name=action.name, requestor=action.requestor, **action.arguments
            )
        except Exception as exc:
            # Upstream EnvironmentEvaluator also continues past invalid reference actions.
            # Some canonical tasks reference removed product IDs in read-only actions.
            gold_action_errors.append({"tool_name": action.name, "error_type": type(exc).__name__})
    live = orchestrator.environment
    match = (
        gold.get_db_hash() == live.get_db_hash()
        and gold.get_user_db_hash() == live.get_user_db_hash()
    )
    checks = [
        EnvAssertionCheck(
            env_assertion=a, met=live.run_env_assertion(a, raise_assertion_error=False), reward=0.0
        )
        for a in criteria.env_assertions or []
    ]
    for check in checks:
        check.reward = float(check.met)
    values = {
        RewardType.DB: float(match),
        RewardType.ENV_ASSERTION: float(all(c.met for c in checks)),
    }
    action_info = None
    if RewardType.ACTION in criteria.reward_basis:
        from tau2.evaluator.evaluator import get_tool_types

        action_info = ActionEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages, tool_types=get_tool_types(live.tools)
        )
        values[RewardType.ACTION] = action_info.reward
    nl_info = None
    if RewardType.NL_ASSERTION in criteria.reward_basis:
        nl_info = NLAssertionsEvaluator.calculate_reward(
            task=task, full_trajectory=simulation.messages
        )
        values[RewardType.NL_ASSERTION] = nl_info.reward
    # Unsupported bases are rejected rather than silently assigned success.
    if set(criteria.reward_basis) - values.keys():
        raise ValueError("Unsupported custom fault reward basis")
    reward = 1.0
    for basis in criteria.reward_basis:
        reward *= values[basis]
    simulation.reward_info = RewardInfo(
        reward=reward,
        db_check=DBCheck(db_match=match, db_reward=float(match)),
        env_assertions=checks,
        action_checks=action_info.action_checks if action_info else None,
        nl_assertions=nl_info.nl_assertions if nl_info else None,
        reward_basis=criteria.reward_basis,
        reward_breakdown={b: values[b] for b in criteria.reward_basis},
        info={
            "evaluator": "custom_live_backend",
            "trajectory_replay": False,
            "gold_action_errors": gold_action_errors,
        },
    )
    return simulation
