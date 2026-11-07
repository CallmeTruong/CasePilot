from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from casepilot.models import ExperimentRun, MetricRecord


def test_unknown_metric_requires_reason():
    with pytest.raises(ValidationError, match="undefined_reason"):
        MetricRecord(value=None, unit="USD")
    assert MetricRecord(value=None, unit="USD", undefined_reason="missing pricing").value is None


@pytest.mark.parametrize(
    "values",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": 0.5, "numerator": 1},
        {"value": 0, "numerator": 0, "denominator": 0},
        {"value": 1, "undefined_reason": "contradiction"},
    ],
)
def test_invalid_metric(values):
    with pytest.raises(ValidationError):
        MetricRecord(unit="ratio", **values)


def run_values():
    return dict(
        experiment_id="E1",
        config_id="single",
        run_id="run",
        config_hash="hash",
        benchmark_revision="a" * 40,
        policy_hash="policy",
        tool_hash="tools",
        dependency_lock_hash="lock",
        model="provider/model",
        user_model="provider/user",
        task_id="0",
        trial=1,
        seed=42,
        episode_id="episode",
    )


def test_run_provenance_and_utc():
    run = ExperimentRun(**run_values())
    assert run.started_at.utcoffset().total_seconds() == 0
    assert run.reward is None
    assert "reward" not in run.agent_context()
    assert "task_id" not in run.agent_context()
    assert run.agent_context() == {"episode_id": "episode"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"trial": 0},
        {"episode_id": " "},
        {"started_at": datetime(2026, 1, 1)},
        {"status": "completed"},
        {"finished_at": datetime(2000, 1, 1, tzinfo=UTC)},
    ],
)
def test_bad_run_provenance(overrides):
    with pytest.raises(ValidationError):
        ExperimentRun(**(run_values() | overrides))
