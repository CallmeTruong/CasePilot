"""Integrity verification and public Retail API inspection. No model requests."""

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from casepilot.config import NonEmpty, StrictModel

Commit = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class BenchmarkManifest(StrictModel):
    repository: Literal["https://github.com/sierra-research/tau2-bench"]
    revision: Commit
    package: str = "tau2==2.0.0"
    requires_python: str = ">=3.12,<3.14"
    domain: Literal["retail"] = "retail"
    source_directory: NonEmpty
    source_files: NonEmpty
    source_files_hash: NonEmpty
    policy_path: NonEmpty
    tools_path: NonEmpty
    split_path: NonEmpty
    policy_hash: NonEmpty
    tool_source_hash: NonEmpty
    task_sets: dict[Literal["development", "final"], list[NonEmpty]] = Field(min_length=2)


def load_manifest(path: Path | str) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Pinned file path escapes source root")
    return target


def verify_source(manifest: BenchmarkManifest, manifest_path: Path | str) -> Path:
    directory = Path(manifest_path).resolve().parent
    source = Path(
        os.environ.get("CASEPILOT_BENCHMARK_PATH") or directory / manifest.source_directory
    ).resolve()
    index_path = _within(directory, manifest.source_files)
    if sha256_file(index_path) != manifest.source_files_hash:
        raise ValueError("Pinned source index integrity mismatch")
    files = json.loads(index_path.read_text(encoding="utf-8"))
    if not files:
        raise ValueError("Empty source index")
    for relative, expected in files.items():
        path = _within(source, relative)
        if not path.is_file():
            raise ValueError(
                f"Missing pinned source {relative}; run python scripts/bootstrap_retail.py"
            )
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if actual != expected:
            raise ValueError(f"Pinned source integrity mismatch: {relative}")
    for relative, expected in (
        (manifest.policy_path, manifest.policy_hash),
        (manifest.tools_path, manifest.tool_source_hash),
    ):
        if sha256_file(_within(source, relative)) != expected:
            raise ValueError(f"Retail content integrity mismatch: {relative}")
    splits = json.loads(_within(source, manifest.split_path).read_text(encoding="utf-8"))
    for name, upstream in (("development", "train"), ("final", "test")):
        tasks = manifest.task_sets.get(name, [])
        if not tasks or len(set(tasks)) != len(tasks) or not set(tasks) <= set(splits[upstream]):
            raise ValueError(f"Task split mismatch for {name}")
    if set(manifest.task_sets["development"]) & set(manifest.task_sets["final"]):
        raise ValueError("Development/final split overlap")
    return source


def prepare_runtime(manifest: BenchmarkManifest, manifest_path: Path | str) -> Path:
    source = verify_source(manifest, manifest_path)
    os.environ["TAU2_DATA_DIR"] = str(source / "data")
    try:
        module = importlib.import_module("tau2")
    except ImportError as exc:
        raise ValueError(
            "Retail package unavailable; run uv sync --locked after bootstrap"
        ) from exc
    expected = source / "src" / "tau2"
    if not Path(module.__file__).resolve().is_relative_to(expected.resolve()):
        raise ValueError("Installed tau2 does not match the verified source directory")
    return source


def inspect_retail(manifest_path: Path | str) -> dict:
    manifest = load_manifest(manifest_path)
    prepare_runtime(manifest, manifest_path)
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.domains.retail.environment import get_environment, get_tasks
    from tau2.environment.environment import Environment
    from tau2.runner.simulation import run_simulation

    environment = get_environment()
    schemas = [tool.openai_schema for tool in sorted(environment.get_tools(), key=lambda t: t.name)]
    # JSON-compatible schemas, not bound functions or database content.
    payload = json.dumps(schemas, sort_keys=True, separators=(",", ":"))
    inventory = {
        "HalfDuplexAgent.__init__": str(inspect.signature(HalfDuplexAgent.__init__)),
        "HalfDuplexAgent.get_init_state": str(inspect.signature(HalfDuplexAgent.get_init_state)),
        "HalfDuplexAgent.generate_next_message": str(
            inspect.signature(HalfDuplexAgent.generate_next_message)
        ),
        "Environment.get_tools": str(inspect.signature(Environment.get_tools)),
        "Environment.get_policy": str(inspect.signature(Environment.get_policy)),
        "Environment.use_tool": str(inspect.signature(Environment.use_tool)),
        "run_simulation": str(inspect.signature(run_simulation)),
    }
    known = {task.id for task in get_tasks(task_split_name=None)}
    if not all(set(ids) <= known for ids in manifest.task_sets.values()):
        raise ValueError("Manifest contains unknown Retail tasks")
    return {
        "repository": manifest.repository,
        "revision": manifest.revision,
        "policy_hash": manifest.policy_hash,
        "tool_source_hash": manifest.tool_source_hash,
        "tool_schema_hash": hashlib.sha256(payload.encode()).hexdigest(),
        "task_sets": manifest.task_sets,
        "api_inventory": inventory,
        "tool_schemas": schemas,
        "validation": "public_api_and_source_integrity_only",
        "live_baseline_validated": False,
    }
