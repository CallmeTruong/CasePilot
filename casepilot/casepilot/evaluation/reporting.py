"""Read explicit experiment collections, retaining raw provenance and missing values."""

import csv
import json
from pathlib import Path

from casepilot.evaluation.aggregate import cluster_bootstrap, describe, paired_deltas


def load_rows(root="results"):
    root = Path(root)
    index = json.loads(
        Path("configs/experiments/completed-runs.json").read_text(encoding="utf-8-sig")
    )
    selected = []
    for experiment, entry in index.items():
        for run in entry["runs"]:
            matches = list(root.glob(f"*/{run}/runs.json"))
            selected.extend((experiment, path.parent) for path in matches)
    attempts = []
    latest = {}
    for collection in ("E2-end-to-end", "E3", "E3-controls", "E4"):
        for path in (root / collection).glob("*/matrix-run.json"):
            matrix = json.loads(path.read_text(encoding="utf-8-sig"))
            declaration = matrix.get("matrix", {})
            variants = (
                ["memory_pair"]
                if declaration.get("memory_dataset")
                else [v["config_id"] for v in declaration.get("variants", [])]
            )
            candidate = (matrix["started_at"], matrix["batch_id"])
            for config_id in variants:
                if not declaration.get("fault_dataset"):
                    key = (matrix["experiment_id"], config_id, None)
                    latest[key] = max(latest.get(key, candidate), candidate)
            for folder in path.parent.iterdir():
                if folder.is_dir() and (
                    (folder / "effective-config.json").exists()
                    or (folder / "memory-results.json").exists()
                ):
                    if (folder / "effective-config.json").exists():
                        config = json.loads(
                            (folder / "effective-config.json").read_text(encoding="utf-8-sig")
                        )
                        config_id = config["config_id"]
                        scenario_id = config.get("scenario_id")
                    else:
                        config_id = "memory_pair"
                        scenario_id = None
                    attempts.append(
                        {
                            "experiment": matrix["experiment_id"],
                            "config_id": config_id,
                            "scenario_id": scenario_id,
                            "batch_id": matrix["batch_id"],
                            "started_at": matrix["started_at"],
                            "folder": folder,
                        }
                    )
    for attempt in attempts:
        key = (attempt["experiment"], attempt["config_id"], attempt["scenario_id"])
        candidate = (attempt["started_at"], attempt["batch_id"])
        latest[key] = max(latest.get(key, candidate), candidate)
    previous = {}
    for attempt in attempts:
        experiment = attempt["experiment"]
        key = (experiment, attempt["config_id"], attempt["scenario_id"])
        if (attempt["started_at"], attempt["batch_id"]) != latest[key]:
            experiment += "-previous"
            previous[str(attempt["folder"])] = attempt["batch_id"][:8]
        selected.append((experiment, attempt["folder"]))
    rows = []
    for experiment, folder in selected:
        memory = folder / "memory-results.json"
        if memory.exists():
            for row in json.loads(memory.read_text(encoding="utf-8-sig")):
                rows.append(
                    row
                    | {
                        "config_id": row["memory"]
                        + ("@" + previous[str(folder)] if str(folder) in previous else ""),
                        "experiment_id": experiment,
                        "reward": float(row["cross_session_success"]),
                        "task_id": row["scenario_id"],
                        "source": str(memory),
                    }
                )
            continue
        if not (folder / "runs.json").exists():
            continue
        config = json.loads((folder / "effective-config.json").read_text(encoding="utf-8-sig"))
        schedule = json.loads((folder / "schedule.json").read_text(encoding="utf-8-sig"))
        raw = json.loads((folder / "runs.json").read_text(encoding="utf-8-sig"))
        observed = {(r["task_id"], r["trial"]) for r in raw}
        for item in schedule:
            if (item["task_id"], item["trial"]) not in observed:
                raw.append(
                    item
                    | {
                        "reward": None,
                        "status": "not_run",
                        "experiment_id": config["experiment_id"],
                        "config_id": config["config_id"],
                        "track": config["track"],
                        "split": config["split"],
                        "run_id": folder.name,
                        "model": config["model"],
                    }
                )
        for row in raw:
            flat = {
                k: v
                for k, v in row.items()
                if k not in {"metrics", "artifact_paths", "inference_parameters"}
            }
            flat.update({k: v["value"] for k, v in row.get("metrics", {}).items()})
            flat.update(
                analysis_experiment=experiment,
                raw_experiment=row["experiment_id"],
                experiment_id=experiment,
                source=str(folder / "runs.json"),
                scheduled=len(schedule),
                architecture=config["architecture"],
                tool_backend=config["tool_backend"],
                policy_access=config["policy_access"],
                recovery_strategy=config.get("recovery_strategy"),
                scenario_id=config.get("scenario_id"),
            )
            if str(folder) in previous:
                flat["config_id"] += "@" + previous[str(folder)]
            rows.append(flat)
    return rows


def summaries(rows):
    groups = {}
    for row in rows:
        key = (row["experiment_id"], row["config_id"], row["track"], row.get("split", "synthetic"))
        groups.setdefault(key, []).append(row)
    result = []
    for (experiment, config, track, split), group in sorted(groups.items()):
        result.append(
            {
                "experiment_id": experiment,
                "config_id": config,
                "track": track,
                "split": split,
                "observations": len(group),
                "success": describe([r.get("reward") for r in group]),
                "ci95": cluster_bootstrap(group),
                "latency_ms": describe([r.get("latency_ms") for r in group]),
                "input_tokens": describe([r.get("input_tokens") for r in group]),
                "output_tokens": describe([r.get("output_tokens") for r in group]),
                "cost": describe([r.get("cost") for r in group]),
                "infrastructure_errors": sum(
                    r.get("status") == "infrastructure_error" for r in group
                ),
            }
        )
    return result


def csv_text(rows):
    import io

    stream = io.StringIO(newline="")
    keys = sorted({key for row in rows for key in row})
    if keys:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return stream.getvalue()


def comparisons(rows):
    result = {}
    for experiment, baseline, keys in (
        ("E1", "casepilot_single_native", ("task_id", "seed", "trial")),
        ("E2", "full_policy", ("task_id", "seed", "trial")),
        ("E4", "M0", ("scenario_id", "trial")),
    ):
        subset = [r for r in rows if r["experiment_id"] == experiment]
        left = [r for r in subset if r["config_id"] == baseline]
        for variant in sorted({r["config_id"] for r in subset} - {baseline}):
            right = [r for r in subset if r["config_id"] == variant]
            result[f"{experiment}:{baseline}->{variant}"] = paired_deltas(left, right, keys=keys)
    return result


def generate(root="results", output="results/reports"):
    rows = load_rows(root)
    summary = summaries(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "episodes.csv").write_text(csv_text(rows), encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output / "paired-comparisons.json").write_text(
        json.dumps(comparisons(rows), indent=2, allow_nan=False), encoding="utf-8"
    )
    from casepilot.evaluation.recovery_metrics import audit_target_refresh, summarize_recovery

    recovery = {}
    for strategy in ("S0", "S1", "S2"):
        fault_rows = [r for r in rows if r["experiment_id"] == "E3" and r["config_id"] == strategy]
        controls = {
            (r["task_id"], r["seed"]): r
            for r in rows
            if r["experiment_id"] == "E3-control" and r["config_id"] == strategy + "_control"
        }
        artifacts, rewards, latencies = [], [], []
        for row in fault_rows:
            folder = Path(row["source"]).parent
            path = folder / f"{row.get('episode_id')}.faults.json"
            if path.exists():
                operations_path = folder / f"{row.get('episode_id')}.operations.json"
                conversation_path = folder / f"{row.get('episode_id')}.conversation.json"
                artifact = json.loads(path.read_text(encoding="utf-8-sig"))
                artifacts.append(
                    audit_target_refresh(
                        artifact,
                        json.loads(operations_path.read_text(encoding="utf-8-sig"))
                        if operations_path.exists()
                        else [],
                        json.loads(conversation_path.read_text(encoding="utf-8-sig"))
                        if conversation_path.exists()
                        else [],
                    )
                )
            control = controls.get((row["task_id"], row["seed"]))
            if control and control.get("reward") is not None and row.get("reward") is not None:
                rewards.append((control["reward"], row["reward"]))
                if control.get("latency_ms") is not None and row.get("latency_ms") is not None:
                    latencies.append((control["latency_ms"], row["latency_ms"]))
        values = summarize_recovery(artifacts, paired_rewards=rewards, paired_latency=latencies)
        recovery[strategy] = {
            k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in values.items()
        }
    (output / "recovery.json").write_text(json.dumps(recovery, indent=2), encoding="utf-8")
    lines = [
        "# Development experiment report",
        "",
        "Custom track; GPT-6 Luna replaces the unavailable upstream judge. No held-out final runs.",
        "Cluster bootstrap: 1,000 resamples, seed 42. Small task counts limit inference.",
        "Agent token usage excludes the user simulator. Endpoint costs are unknown.",
        "Latest launched attempt per variant/scenario is primary, independent of outcome.",
        "Earlier attempts remain in separate *-previous cohorts; nothing is selected by score.",
        "E1 groups legacy phase0 labels; original artifacts are unchanged.",
        "",
        "| Experiment | Config | Evaluated / scheduled | Mean success | 95% CI | Latency ms |",
        "|---|---|---:|---:|---|---:|",
    ]
    for row in summary:
        ci = row["ci95"]
        bounds = "undefined" if ci["low"] is None else f"{ci['low']:.3f}–{ci['high']:.3f}"
        mean = row["success"]["mean"]
        latency = row["latency_ms"]["mean"]
        lines.append(
            f"| {row['experiment_id']} | {row['config_id']} | "
            f"{row['success']['n']} / {row['observations']} | "
            f"{mean if mean is not None else 'unknown'} | {bounds} | "
            f"{round(latency, 1) if latency is not None else 'unknown'} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if summary:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plotted = [
            s
            for s in summary
            if s["success"]["mean"] is not None and not s["experiment_id"].endswith("-previous")
        ]
        labels = [f"{s['experiment_id']}:{s['config_id']}" for s in plotted]
        fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.35)))
        ax.barh(labels, [s["success"]["mean"] for s in plotted])
        ax.set(xlim=(0, 1), xlabel="Observed success fraction (custom development)")
        fig.tight_layout()
        fig.savefig(output / "success.png", dpi=180)
        plt.close(fig)
    return output
