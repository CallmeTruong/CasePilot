import pytest
from pydantic import ValidationError

from casepilot.agents.executor import ExecutionDecision
from casepilot.agents.planner import ResolutionPlan


def test_planner_schema_does_not_execute_actions():
    plan = ResolutionPlan(
        intent="lookup",
        required_information=[],
        evidence_ids=[],
        actions=["verify identity"],
        requires_confirmation=False,
        status="ready",
    )
    assert plan.actions == ["verify identity"]
    with pytest.raises(ValidationError):
        ResolutionPlan.model_validate(plan.model_dump() | {"status": "pretend_success"})


def test_expected_result_is_not_observed_result():
    decision = ExecutionDecision(
        action="lookup",
        tool_name=None,
        arguments={},
        requires_confirmation=False,
        expected_result="needs evidence",
    )
    assert "observed_result" not in decision.model_dump()
