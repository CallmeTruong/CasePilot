import json
from pathlib import Path

import pytest

from casepilot.config import load_config
from casepilot.evaluation import memory_runner


def test_source_session_failure_closes_store_and_preserves_folder(monkeypatch, tmp_path):
    closed = []
    original = memory_runner.HistoryStore

    class RecordingStore(original):
        def close(self):
            closed.append(True)
            super().close()

    def fail(*_args, **_kwargs):
        raise RuntimeError("test-only provider failure")

    monkeypatch.setattr(memory_runner, "HistoryStore", RecordingStore)
    monkeypatch.setattr(memory_runner, "_session", fail)
    monkeypatch.setattr(memory_runner, "load_provider_settings", lambda _: None)
    scenario = json.loads(Path("datasets/memory_cases/scenarios.jsonl").read_text().splitlines()[0])
    with pytest.raises(RuntimeError) as captured:
        memory_runner.run_memory_scenario(
            "configs/benchmark-manifest.json",
            load_config("configs/default.yaml").model_dump(),
            scenario,
            trial=1,
            output=tmp_path,
        )
    assert closed == [True]
    assert Path(captured.value.artifact_folder).is_dir()
    folder = Path(captured.value.artifact_folder)
    assert json.loads((folder / "schedule.json").read_text())["modes"] == ["M0", "M1"]
    assert (folder / "effective-configs.json").exists()
    assert (folder / "manifest.json").exists()
