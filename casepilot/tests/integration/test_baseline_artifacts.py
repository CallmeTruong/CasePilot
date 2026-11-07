"""Artifact contract tests with a mocked model run; not benchmark results."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from casepilot.config import ProviderSettings, RunConfig
from casepilot.integration.baseline import run_upstream_baseline
from casepilot.integration.benchmark import load_manifest, prepare_runtime

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/benchmark-manifest.json"
pytestmark = pytest.mark.integration


@pytest.fixture
def simulation_module():
    prepare_runtime(load_manifest(MANIFEST), MANIFEST)
    import tau2.runner.simulation as module

    return module


def test_reward_zero_is_completed_not_infrastructure_failure(
    tmp_path, monkeypatch, simulation_module
):
    reward = SimpleNamespace(reward=0.0, model_dump=lambda **_: {"reward": 0.0})
    simulation = SimpleNamespace(
        reward_info=reward, messages=[], termination_reason=SimpleNamespace(value="agent_stop")
    )
    monkeypatch.setattr(simulation_module, "run_simulation", lambda _: simulation)
    folder = run_upstream_baseline(
        MANIFEST,
        RunConfig(model="gpt-4o", user_model="gpt-4o"),
        ProviderSettings(api_key=SecretStr("test-only-key")),
        tmp_path,
    )
    records = json.loads((folder / "runs.json").read_text())
    assert len(records) == 3
    assert all(r["status"] == "completed" and r["reward"] == 0.0 for r in records)
    assert len({r["episode_id"] for r in records}) == 3
    assert all(r["metrics"]["cost"]["value"] is None for r in records)
    assert "test-only-key" not in (folder / "effective-config.json").read_text()


def test_provider_failure_preserves_schedule_and_partial_metadata(
    tmp_path,
    monkeypatch,
    simulation_module,
):
    def fail(_):
        raise RuntimeError("private provider payload test-only-key")

    monkeypatch.setattr(simulation_module, "run_simulation", fail)
    with pytest.raises(RuntimeError, match="partial run"):
        run_upstream_baseline(
            MANIFEST,
            RunConfig(model="gpt-4o", user_model="gpt-4o"),
            ProviderSettings(api_key=SecretStr("test-only-key")),
            tmp_path,
        )
    folder = next(tmp_path.iterdir())
    assert len(json.loads((folder / "schedule.json").read_text())) == 3
    records = json.loads((folder / "runs.json").read_text())
    assert len(records) == 1
    assert records[0]["status"] == "infrastructure_error" and records[0]["reward"] is None
    assert records[0]["termination_reason"] == "RuntimeError"
    for file in folder.iterdir():
        assert "test-only-key" not in file.read_text()
