import json

from casepilot.evaluation.reporting import comparisons, load_rows


def test_latest_attempt_is_selected_by_start_time_not_reward(tmp_path):
    root = tmp_path / "results"
    for batch_id, started, reward in (
        ("old", "2026-11-05T00:00:00+00:00", 1),
        ("new", "2026-11-06T00:00:00+00:00", 0),
    ):
        batch = root / "E2-end-to-end" / batch_id
        folder = batch / (batch_id + "run")
        folder.mkdir(parents=True)
        (batch / "matrix-run.json").write_text(
            json.dumps(
                {"batch_id": batch_id, "started_at": started, "experiment_id": "E2", "jobs": []}
            )
        )
        config = {
            "config_id": "reranked",
            "experiment_id": "E2",
            "track": "custom",
            "split": "development",
            "architecture": "single",
            "tool_backend": "native",
            "policy_access": "hybrid_rerank",
            "model": "fixture",
        }
        (folder / "effective-config.json").write_text(json.dumps(config))
        (folder / "schedule.json").write_text(
            json.dumps([{"task_id": "0", "trial": 1, "seed": 42}])
        )
        row = {
            "experiment_id": "E2",
            "config_id": "reranked",
            "task_id": "0",
            "trial": 1,
            "seed": 42,
            "track": "custom",
            "split": "development",
            "reward": reward,
            "status": "completed",
        }
        (folder / "runs.json").write_text(json.dumps([row]))
    rows = load_rows(root)
    primary = [r for r in rows if r["experiment_id"] == "E2"]
    previous = [r for r in rows if r["experiment_id"] == "E2-previous"]
    assert len(primary) == len(previous) == 1
    assert primary[0]["reward"] == 0 and previous[0]["reward"] == 1
    assert previous[0]["config_id"] == "reranked@old"
    comparisons(rows)  # Repeated attempts cannot collide as paired observations.


def test_targeted_retry_keeps_other_scenarios_from_original_batch(tmp_path):
    root = tmp_path / "results"
    for batch_id, started, scenarios in (
        ("old", "2026-11-05T00:00:00+00:00", ["A", "B"]),
        ("retry", "2026-11-06T00:00:00+00:00", ["B"]),
    ):
        batch = root / "E3" / batch_id
        batch.mkdir(parents=True)
        (batch / "matrix-run.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "started_at": started,
                    "experiment_id": "E3",
                    "jobs": [],
                    "matrix": {"fault_dataset": "fixture", "variants": [{"config_id": "S2"}]},
                }
            )
        )
        for scenario in scenarios:
            folder = batch / scenario
            folder.mkdir()
            config = {
                "config_id": "S2",
                "experiment_id": "E3",
                "track": "custom",
                "split": "development",
                "architecture": "single",
                "tool_backend": "native",
                "policy_access": "full",
                "model": "fixture",
                "scenario_id": scenario,
            }
            (folder / "effective-config.json").write_text(json.dumps(config))
            (folder / "schedule.json").write_text(
                json.dumps([{"task_id": "0", "trial": 1, "seed": 42}])
            )
            (folder / "runs.json").write_text(
                json.dumps(
                    [
                        {
                            "experiment_id": "E3",
                            "config_id": "S2",
                            "task_id": "0",
                            "trial": 1,
                            "seed": 42,
                            "track": "custom",
                            "reward": 0,
                            "status": "completed",
                        }
                    ]
                )
            )
    rows = load_rows(root)
    primary = [r for r in rows if r["experiment_id"] == "E3"]
    assert sorted(r["scenario_id"] for r in primary) == ["A", "B"]
    assert len([r for r in rows if r["experiment_id"] == "E3-previous"]) == 1
