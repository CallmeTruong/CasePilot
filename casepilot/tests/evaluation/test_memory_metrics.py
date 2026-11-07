from casepilot.evaluation.aggregate import paired_deltas


def test_memory_pairs_by_scenario_and_trial():
    base = {"scenario_id": "synthetic", "trial": 1, "reward": 0}
    result = paired_deltas([base], [base | {"reward": 1}], keys=("scenario_id", "trial"))
    assert result["pairs"] == 1
    assert result["summary"]["mean"] == 1
    assert result["ci95"]["low"] is None
