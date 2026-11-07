import json
from pathlib import Path

import streamlit as st

from casepilot.evaluation.reporting import csv_text, load_rows, summaries


def render():
    batches = []
    for collection in ("E2-end-to-end", "E3", "E3-controls", "E4"):
        matrices = list(Path("results", collection).glob("*/matrix-run.json"))
        if matrices:
            latest = max(
                matrices, key=lambda p: json.loads(p.read_text(encoding="utf-8-sig"))["started_at"]
            )
            matrix = json.loads(latest.read_text(encoding="utf-8-sig"))
            batches.append(
                {
                    "experiment": matrix["experiment_id"],
                    "completed_jobs": sum(j["status"] == "completed" for j in matrix["jobs"]),
                    "failed_jobs": sum(j["status"] == "failed" for j in matrix["jobs"]),
                    "scheduled_jobs": matrix["job_count"],
                    "status": "finished" if matrix.get("finished_at") else "running / interrupted",
                    "batch_id": matrix["batch_id"],
                }
            )
    with st.expander("Latest batch progress"):
        st.dataframe(batches, width="stretch")
    rows = load_rows()
    if not rows:
        st.info("No experiment artifacts yet. Run scripts/run_experiments.py first.")
        return
    if not st.checkbox("Include earlier attempts", value=False):
        rows = [r for r in rows if not r["experiment_id"].endswith("-previous")]
    experiments = st.multiselect(
        "Experiments",
        sorted({r["experiment_id"] for r in rows}),
        default=sorted({r["experiment_id"] for r in rows}),
    )
    selected = [r for r in rows if r["experiment_id"] in experiments]
    summary = summaries(selected)
    st.caption("Custom development · 95% task/scenario-cluster bootstrap · costs unknown")
    st.dataframe(
        [
            {
                "experiment": s["experiment_id"],
                "config": s["config_id"],
                "n": s["observations"],
                "success": s["success"]["mean"],
                "sample_sd": s["success"]["sample_sd"],
                "ci_low": s["ci95"]["low"],
                "ci_high": s["ci95"]["high"],
                "latency_p50_ms": s["latency_ms"]["p50"],
                "latency_p95_ms": s["latency_ms"]["p95"],
                "input_tokens": s["input_tokens"]["mean"],
                "cost": s["cost"]["mean"],
                "infrastructure_errors": s["infrastructure_errors"],
            }
            for s in summary
        ],
        width="stretch",
    )
    st.download_button(
        "Export episodes CSV", csv_text(selected), "casepilot-episodes.csv", "text/csv"
    )
    failures = [r for r in selected if r.get("reward") != 1]
    with st.expander("Errors / unsuccessful episodes"):
        st.dataframe(failures, width="stretch")
    component = Path("results/E2-component.json")
    if component.exists():
        with st.expander("Retrieval component metrics"):
            st.json(json.loads(component.read_text(encoding="utf-8-sig")))
    with st.expander("Episode provenance / trace locations"):
        st.dataframe(
            [
                {
                    k: r.get(k)
                    for k in (
                        "experiment_id",
                        "run_id",
                        "episode_id",
                        "source",
                        "benchmark_revision",
                        "config_hash",
                    )
                }
                for r in selected
            ]
        )
