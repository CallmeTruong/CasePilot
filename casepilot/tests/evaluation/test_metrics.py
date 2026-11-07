from casepilot.evaluation.metrics import cost_per_success, ratio, summarize_runs


def test_scheduled_coverage_is_separate_from_observed_success():
    result = summarize_runs(
        [
            {"reward": 1, "status": "completed"},
            {"reward": None, "status": "infrastructure_error"},
        ],
        scheduled_count=3,
    )
    assert result["task_success"].value == 1
    assert result["confirmed_success_lower_bound"].value == 1 / 3
    assert result["evaluation_coverage"].value == 1 / 3
    assert result["missing_rewards"] == 2
    assert result["infrastructure_failures"] == 1


def test_unknown_cost_and_zero_denominators_remain_unknown():
    assert cost_per_success([None, 1], 1).value is None
    assert cost_per_success([1], 0).value is None
    assert cost_per_success([1, 2], 2).value == 1.5
    assert ratio(0, 0).value is None
