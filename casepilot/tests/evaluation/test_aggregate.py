import pytest

from casepilot.evaluation.aggregate import cluster_bootstrap, describe, paired_deltas


def test_missing_values_and_sample_sd():
    assert describe([None])["mean"] is None
    assert describe([1])["sample_sd"] is None
    assert describe([1, 3, None])["sample_sd"] == pytest.approx(2**0.5)
    assert describe([1, 3])["p95"] == pytest.approx(2.9)


def test_clustered_ci_is_reproducible_and_not_trial_independence():
    rows = [{"task_id": "one", "reward": 1}] * 20
    assert cluster_bootstrap(rows)["low"] is None
    rows += [{"task_id": "two", "reward": 0}]
    assert cluster_bootstrap(rows) == cluster_bootstrap(rows)
    assert cluster_bootstrap(rows)["clusters"] == 2


def test_pairing_excludes_missing_and_rejects_duplicates():
    a = [{"task_id": "a", "seed": 42, "trial": 1, "reward": 0}]
    b = [a[0] | {"reward": 1}]
    assert paired_deltas(a, b)["summary"]["mean"] == 1
    assert paired_deltas(a, [b[0] | {"reward": None}])["pairs"] == 0
    with pytest.raises(ValueError, match="Ambiguous"):
        paired_deltas(a + a, b)
