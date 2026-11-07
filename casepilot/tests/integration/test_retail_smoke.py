"""Upstream evaluator sees the native mutation's end state, with no provider calls."""

import pytest

pytestmark = pytest.mark.integration


def test_evaluator_observes_the_mutated_environment(retail_environment):
    from tau2.data_model.message import AssistantMessage, ToolCall
    from tau2.data_model.tasks import Action, EvaluationCriteria, RewardType
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator

    env = retail_environment
    customer = next(iter(env.tools.db.users))
    args = {
        "user_id": customer,
        "address1": "Synthetic fixture street",
        "address2": "",
        "city": "Test",
        "state": "CA",
        "country": "USA",
        "zip": "90001",
    }
    task = get_tasks("train")[0].model_copy(
        deep=True,
        update={
            "id": "synthetic-evaluator-fixture",
            "initial_state": None,
            "evaluation_criteria": EvaluationCriteria(
                actions=[Action(action_id="expected", name="modify_user_address", arguments=args)],
                reward_basis=[RewardType.DB],
            ),
        },
    )
    call = ToolCall(id="fixture-operation", name="modify_user_address", arguments=args)
    message = AssistantMessage(role="assistant", tool_calls=[call])
    result = env.get_response(call)
    assert not result.error
    observed = EnvironmentEvaluator.calculate_reward(get_environment, task, [message, result])
    missing = EnvironmentEvaluator.calculate_reward(get_environment, task, [])
    assert observed.reward == 1.0 and missing.reward == 0.0
    assert observed.db_check.db_match
