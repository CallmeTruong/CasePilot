import json
from concurrent.futures import ThreadPoolExecutor

import yaml

from casepilot.evaluation import batch


def test_provider_circuit_breaker_preserves_unstarted_schedule(monkeypatch, tmp_path):
    calls = []

    def fail(job):
        calls.append(job)
        return {"status": "failed", "error_type": "APIError"}

    monkeypatch.setattr(batch, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(batch, "_variant_job", fail)
    monkeypatch.setattr(batch, "load_provider_settings", lambda _: None)
    monkeypatch.setattr(batch, "require_provider", lambda *_args: [])
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "fixture",
                "base_config": "configs/default.yaml",
                "variants": [{"config_id": f"fixture_{i}"} for i in range(5)],
            }
        )
    )
    root = batch.run_matrix(
        "configs/benchmark-manifest.json", matrix, output=tmp_path / "results", workers=1
    )
    saved = json.loads((root / "matrix-run.json").read_text())
    assert len(calls) == 3
    assert saved["job_count"] == len(saved["jobs"]) == 5
    assert [r["status"] for r in saved["jobs"]].count("not_run") == 2
    assert "circuit_breaker" in saved


def test_denied_inference_creates_no_batch_and_starts_no_workers(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setattr(batch, "load_provider_settings", lambda _: None)

    def denied(*_args):
        raise ValueError("Provider inference unavailable (chat: 403)")

    def forbidden(**_kwargs):
        raise AssertionError("No worker should start before inference preflight")

    monkeypatch.setattr(batch, "require_provider", denied)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", forbidden)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "fixture",
                "base_config": "configs/default.yaml",
                "variants": [{"config_id": "fixture"}],
            }
        )
    )
    with pytest.raises(ValueError, match="chat: 403"):
        batch.run_matrix("configs/benchmark-manifest.json", matrix, output=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_crashed_worker_preserves_remaining_schedule(monkeypatch, tmp_path):
    from concurrent.futures.process import BrokenProcessPool

    def crashed(_job):
        raise BrokenProcessPool("test-only worker crash")

    monkeypatch.setattr(batch, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(batch, "_variant_job", crashed)
    monkeypatch.setattr(batch, "load_provider_settings", lambda _: None)
    monkeypatch.setattr(batch, "require_provider", lambda *_args: [])
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "fixture",
                "base_config": "configs/default.yaml",
                "variants": [{"config_id": f"fixture_{i}"} for i in range(3)],
            }
        )
    )
    root = batch.run_matrix(
        "configs/benchmark-manifest.json", matrix, output=tmp_path / "results", workers=1
    )
    saved = json.loads((root / "matrix-run.json").read_text())
    assert saved["circuit_breaker"] == "worker_pool_unavailable"
    assert len(saved["jobs"]) == 3
    assert saved["jobs"][0]["config_id"] == "fixture_0"
    assert [r["status"] for r in saved["jobs"]] == ["failed", "not_run", "not_run"]
