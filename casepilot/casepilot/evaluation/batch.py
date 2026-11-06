import json
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path

import yaml

from casepilot.config import RunConfig, load_config, load_provider_settings
from casepilot.integration.provider_health import require_provider
from casepilot.integration.simulation_runner import run_casepilot


def _variant_job(job):
    try:
        folder = run_casepilot(
            Path(job["manifest"]),
            RunConfig.model_validate(job["config"]),
            load_provider_settings(job["env_file"]),
            Path(job["output"]),
        )
        return {
            "status": "completed",
            "folder": str(folder),
            "config_id": job["config"]["config_id"],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "folder": getattr(exc, "artifact_folder", None),
            "config_id": job["config"]["config_id"],
            "scenario_id": job["config"].get("scenario_id"),
        }


def _memory_job(job):
    from casepilot.evaluation.memory_runner import run_memory_scenario

    try:
        folder = run_memory_scenario(
            job["manifest"],
            job["base_values"],
            job["scenario"],
            trial=job["trial"],
            output=job["output"],
            env_file=job["env_file"],
        )
        return {
            "status": "completed",
            "folder": folder,
            "scenario_id": job["scenario"]["scenario_id"],
            "trial": job["trial"],
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "folder": getattr(exc, "artifact_folder", None),
            "scenario_id": job["scenario"]["scenario_id"],
            "trial": job["trial"],
        }


def run_matrix(manifest_path, matrix_path, *, output, env_file=".env", trials=None, workers=1):
    if not 1 <= workers <= 4:
        raise ValueError("workers must be 1..4")
    matrix = yaml.safe_load(Path(matrix_path).read_text(encoding="utf-8"))
    base = load_config(matrix["base_config"]).model_dump()
    count = trials if trials is not None else matrix.get("trials", 3)
    base.update(
        experiment_id=matrix["experiment_id"], track=matrix.get("track", "custom"), trials=count
    )
    for key, value in matrix.get("controls", {}).items():
        if key in RunConfig.model_fields:
            base[key] = value
        elif key == "manifest" and Path(value).resolve() != Path(manifest_path).resolve():
            raise ValueError("Matrix and CLI benchmark manifests differ")
        elif key != "manifest":
            raise ValueError("Unknown matrix control")
    RunConfig.model_validate(base)
    settings = load_provider_settings(env_file)
    batch_id = uuid.uuid4().hex
    root = Path(output).resolve() / batch_id
    common = {
        "manifest": str(Path(manifest_path).resolve()),
        "env_file": str(Path(env_file).resolve()),
        "output": str(root),
    }
    jobs = []
    control_keys = set()
    worker = _variant_job
    if matrix.get("memory_dataset"):
        worker = _memory_job
        scenarios = [
            json.loads(line)
            for line in Path(matrix["memory_dataset"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for scenario in scenarios:
            for trial in range(1, count + 1):
                jobs.append(
                    common
                    | {
                        "base_values": base | {"seed": base["seed"] + trial - 1},
                        "scenario": scenario,
                        "trial": trial,
                    }
                )
    else:
        scenarios = [None]
        if matrix.get("fault_dataset"):
            scenarios = [
                json.loads(line)
                for line in Path(matrix["fault_dataset"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if matrix.get("scenario_ids") is not None:
                requested = matrix["scenario_ids"]
                known = {scenario["scenario_id"] for scenario in scenarios}
                if (
                    not requested
                    or len(set(requested)) != len(requested)
                    or not set(requested) <= known
                ):
                    raise ValueError(
                        "scenario_ids must be a nonempty unique subset of the fault dataset"
                    )
                scenarios = [
                    scenario for scenario in scenarios if scenario["scenario_id"] in requested
                ]
        for variant in matrix["variants"]:
            for scenario in scenarios:
                values = base | variant
                if scenario:
                    is_fault = variant.get("fault_injection", True)
                    control_key = (variant["config_id"], scenario["task_id"], scenario["seed"])
                    if not is_fault and control_key in control_keys:
                        continue
                    control_keys.add(control_key)
                    values.update(
                        scenario_id=scenario["scenario_id"],
                        task_ids=[scenario["task_id"]],
                        fault_injection=is_fault,
                        fault_type=scenario["fault_type"] if is_fault else None,
                        fault_target=scenario["tool_target"] if is_fault else None,
                        seed=scenario["seed"],
                        trials=count,
                    )
                validated = RunConfig.model_validate(values)
                jobs.append(common | {"config": validated.model_dump()})
    configs = [RunConfig.model_validate(job.get("config", job.get("base_values"))) for job in jobs]
    health = require_provider(settings, configs)
    root.mkdir(parents=True, exist_ok=False)
    summary = {
        "provider_preflight": health,
        "batch_id": batch_id,
        "experiment_id": matrix["experiment_id"],
        "started_at": datetime.now(UTC).isoformat(),
        "workers": workers,
        "matrix": matrix,
        "job_count": len(jobs),
        "jobs": [],
    }

    def save():
        (root / "matrix-run.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    save()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        remaining = iter(jobs)
        futures = {
            pool.submit(worker, job): job
            for job in [next(remaining, None) for _ in range(workers)]
            if job is not None
        }
        consecutive_failures = 0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "failed", "error_type": type(exc).__name__}
                    if isinstance(exc, BrokenProcessPool):
                        summary["circuit_breaker"] = "worker_pool_unavailable"
                result.setdefault("config_id", job.get("config", {}).get("config_id"))
                result.setdefault("scenario_id", job.get("config", {}).get("scenario_id"))
                consecutive_failures = (
                    consecutive_failures + 1 if result["status"] == "failed" else 0
                )
                summary["jobs"].append(result)
                if consecutive_failures >= 3 and "circuit_breaker" not in summary:
                    summary["circuit_breaker"] = "three consecutive infrastructure failures"
                save()
                print(
                    f"{matrix['experiment_id']}: {len(summary['jobs'])}/{len(jobs)} "
                    f"jobs {result['status']}",
                    flush=True,
                )
                if "circuit_breaker" not in summary:
                    following = next(remaining, None)
                    if following is not None:
                        futures[pool.submit(worker, following)] = following
        for job in remaining:
            summary["jobs"].append(
                {
                    "status": "not_run",
                    "reason": summary.get("circuit_breaker", "batch_stopped"),
                    "config_id": job.get("config", {}).get("config_id"),
                    "scenario_id": job.get("scenario", {}).get("scenario_id"),
                }
            )
    summary["finished_at"] = datetime.now(UTC).isoformat()
    save()
    return root
