import hashlib
import json
from pathlib import Path

import pytest

from casepilot.integration.benchmark import load_manifest, verify_source


def make_snapshot(tmp_path):
    source = tmp_path / "dependency"
    source.mkdir()
    (source / "policy.md").write_text("policy")
    (source / "tools.py").write_text("tools")
    (source / "split.json").write_text('{"train": ["0", "1", "2"], "test": ["3"]}')
    hashes = {}
    for name in ("policy.md", "tools.py", "split.json"):
        data = (source / name).read_bytes()
        hashes[name] = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    index = tmp_path / "files.json"
    index.write_text(json.dumps(hashes))
    manifest = {
        "repository": "https://github.com/sierra-research/tau2-bench",
        "revision": "a" * 40,
        "source_directory": "dependency",
        "source_files": "files.json",
        "source_files_hash": hashlib.sha256(index.read_bytes()).hexdigest(),
        "policy_path": "policy.md",
        "tools_path": "tools.py",
        "split_path": "split.json",
        "policy_hash": hashlib.sha256((source / "policy.md").read_bytes()).hexdigest(),
        "tool_source_hash": hashlib.sha256((source / "tools.py").read_bytes()).hexdigest(),
        "task_sets": {"development": ["0", "1", "2"], "final": ["3"]},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, source


def test_snapshot_requires_every_pinned_blob(tmp_path, monkeypatch):
    monkeypatch.delenv("CASEPILOT_BENCHMARK_PATH", raising=False)
    path, source = make_snapshot(tmp_path)
    manifest = load_manifest(path)
    assert verify_source(manifest, path) == source
    (source / "tools.py").write_text("changed upstream")
    with pytest.raises(ValueError, match="integrity"):
        verify_source(manifest, path)


def test_missing_dependency_actionable(tmp_path, monkeypatch):
    monkeypatch.delenv("CASEPILOT_BENCHMARK_PATH", raising=False)
    path, source = make_snapshot(tmp_path)
    (source / "policy.md").unlink()
    with pytest.raises(ValueError, match="bootstrap_retail"):
        verify_source(load_manifest(path), path)


def test_source_index_tamper_rejected(tmp_path):
    path, _ = make_snapshot(tmp_path)
    (tmp_path / "files.json").write_text("{}")
    with pytest.raises(ValueError, match="index"):
        verify_source(load_manifest(path), path)


def test_task_split_leakage_rejected(tmp_path):
    path, _ = make_snapshot(tmp_path)
    data = json.loads(path.read_text())
    data["task_sets"]["development"] = ["3"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="split"):
        verify_source(load_manifest(path), path)


def test_path_traversal_rejected(tmp_path):
    path, _ = make_snapshot(tmp_path)
    index = tmp_path / "files.json"
    index.write_text('{"../outside": "bad"}')
    data = json.loads(path.read_text())
    data["source_files_hash"] = hashlib.sha256(index.read_bytes()).hexdigest()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        verify_source(load_manifest(path), path)


def test_manifest_pinned_revision_required(tmp_path):
    path, _ = make_snapshot(tmp_path)
    data = json.loads(path.read_text())
    data["revision"] = "main"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_checked_in_manifest_has_no_absolute_paths():
    path = Path(__file__).resolve().parents[2] / "configs/benchmark-manifest.json"
    manifest = load_manifest(path)
    assert not Path(manifest.source_directory).is_absolute()
