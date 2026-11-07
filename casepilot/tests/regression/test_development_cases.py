import json
from pathlib import Path

import pytest

from casepilot.integration.benchmark import load_manifest


def test_development_regression_set_is_frozen_and_not_held_out():
    root = Path(__file__).resolve().parents[2]
    frozen = json.loads((Path(__file__).parent / "development_manifest.json").read_text())
    manifest = load_manifest(root / "configs/benchmark-manifest.json")
    assert frozen["benchmark_revision"] == manifest.revision
    assert frozen["task_ids"] == manifest.task_sets["development"]
    assert not set(frozen["task_ids"]) & set(manifest.task_sets["final"])


@pytest.mark.live
def test_live_development_cases_complete_with_evaluator(tmp_path):
    from casepilot.config import load_config, load_provider_settings
    from casepilot.integration.simulation_runner import run_casepilot

    root = Path(__file__).resolve().parents[2]
    folder = run_casepilot(
        root / "configs/benchmark-manifest.json",
        load_config(root / "configs/default.yaml"),
        load_provider_settings(root / ".env"),
        tmp_path,
    )
    records = json.loads((folder / "runs.json").read_text(encoding="utf-8"))
    assert [r["task_id"] for r in records] == ["0", "1", "2"]
    assert all(r["status"] == "completed" and r["reward"] is not None for r in records)
    assert all(r["termination_reason"] != "agent_error" for r in records)
