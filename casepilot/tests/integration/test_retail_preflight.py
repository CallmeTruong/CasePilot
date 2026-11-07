"""Offline public-API preflight, not the live baseline/evaluator gate T009/T010."""

from pathlib import Path

import pytest

from casepilot.config import RunConfig
from casepilot.integration.baseline import make_upstream_orchestrator
from casepilot.integration.benchmark import inspect_retail, load_manifest, prepare_runtime

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/benchmark-manifest.json"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def retail():
    prepare_runtime(load_manifest(MANIFEST), MANIFEST)
    from tau2.domains.retail.environment import get_environment, get_tasks

    return get_environment, get_tasks


def test_actual_inventory_and_split_membership(retail):
    inventory = inspect_retail(MANIFEST)
    assert len(inventory["tool_schemas"]) == 16
    assert inventory["live_baseline_validated"] is False
    assert "Environment.use_tool" in inventory["api_inventory"]
    assert set(inventory["task_sets"]["development"]) == {"0", "1", "2"}
    assert inventory["tool_schema_hash"] == (
        "669f1a85b06f19543db913f1ee455bac9cda2f57ff96c66c58d8ad795480813e"
    )


def test_environment_instances_are_isolated(retail):
    get_environment, _ = retail
    first, second = get_environment(), get_environment()
    before = second.get_db_hash()
    customer = next(iter(first.tools.db.users))
    original = first.use_tool("get_user_details", user_id=customer).model_copy(deep=True)
    first.use_tool(
        "modify_user_address",
        user_id=customer,
        address1="Offline fixture street",
        address2="",
        city="Test",
        state="CA",
        country="USA",
        zip="90001",
    )
    assert first.get_db_hash() != before
    assert second.get_db_hash() == before
    assert second.use_tool("get_user_details", user_id=customer).address == original.address


def test_upstream_orchestrator_can_be_built_without_provider_call(retail):
    _, get_tasks = retail
    task = next(t for t in get_tasks("train") if t.id == "0")
    config = RunConfig(model="gpt-4o", user_model="gpt-4o")
    orchestrator = make_upstream_orchestrator(config, task, seed=42, simulation_id="offline-test")
    assert orchestrator.agent.llm == "gpt-4o"
    assert orchestrator.user.llm == "gpt-4o"
    assert str(task.user_scenario) not in orchestrator.agent.system_prompt
    assert orchestrator.environment.get_domain_name() == "retail"
    assert orchestrator.agent.get_init_state() is not None
    assert orchestrator.user.get_init_state() is not None
