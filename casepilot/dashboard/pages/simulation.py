import json
from pathlib import Path

import streamlit as st

from casepilot.config import RunConfig, load_config, load_provider_settings
from casepilot.integration.benchmark import load_manifest
from casepilot.integration.simulation_runner import run_casepilot


def show_run(folder):
    folder = Path(folder)
    records = json.loads((folder / "runs.json").read_text(encoding="utf-8-sig"))
    st.dataframe(records, width="stretch")
    episode = st.selectbox("Episode", [r["episode_id"] for r in records])
    record = next(r for r in records if r["episode_id"] == episode)
    columns = st.columns(3)
    columns[0].metric(
        "Evaluator reward", record.get("reward") if record.get("reward") is not None else "unknown"
    )
    columns[1].metric("Status", record["status"])
    columns[2].metric("Termination", record.get("termination_reason") or "unknown")
    for label, relative in record.get("artifact_paths", {}).items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            continue
        with st.expander(label):
            if path.suffix == ".jsonl":
                st.json(
                    [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
                )
            else:
                st.json(json.loads(path.read_text(encoding="utf-8")))


def render():
    base = load_config("configs/default.yaml")
    manifest = load_manifest("configs/benchmark-manifest.json")
    st.caption("Run a development task with API calls. Keys and the base URL are loaded from .env.")
    columns = st.columns(4)
    task = columns[0].selectbox("Task", manifest.task_sets["development"])
    architecture = columns[1].selectbox("Architecture", ["single", "multi"])
    backend = columns[2].selectbox("Tools", ["native", "mcp"])
    policy = columns[3].selectbox("Policy", ["full", "bm25", "dense", "hybrid", "hybrid_rerank"])
    fault = st.selectbox("Custom fault", ["none", "F1", "F2", "F3"])
    target = None
    if fault != "none":
        target = st.selectbox(
            "Fault target", ["exchange_delivered_order_items", "return_delivered_order_items"]
        )
    st.caption(f"Model: {base.model} · Judge: {base.judge_model} · Custom track")
    with st.expander("Saved runs — inspect without API calls"):
        saved = sorted(
            Path("results").glob("**/runs.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if saved:
            paths = [str(path.parent) for path in saved]
            requested = st.query_params.get("saved_run")
            if requested in paths and requested != st.session_state.get("_last_saved_run_query"):
                st.session_state["simulation_folder"] = requested
                st.session_state["_last_saved_run_query"] = requested
            chosen = st.selectbox(
                "Saved run", paths, index=paths.index(requested) if requested in paths else 0
            )
            if st.button("Open saved run"):
                st.session_state["simulation_folder"] = chosen
        else:
            st.caption("No saved runs yet.")
    if st.button("Run simulation", type="primary"):
        try:
            config = RunConfig.model_validate(
                base.model_dump()
                | {
                    "experiment_id": "demo",
                    "config_id": "dashboard",
                    "track": "custom",
                    "architecture": architecture,
                    "tool_backend": backend,
                    "policy_access": policy,
                    "task_ids": [task],
                    "trials": 1,
                    "fault_injection": fault != "none",
                    "fault_type": None if fault == "none" else fault,
                    "fault_target": target,
                }
            )
            with st.spinner("Agent and evaluator are running…"):
                folder = run_casepilot(
                    Path("configs/benchmark-manifest.json"),
                    config,
                    load_provider_settings(),
                    Path("results/dashboard"),
                )
            st.session_state["simulation_folder"] = str(folder)
        except Exception as exc:
            st.error(
                f"Simulation failed ({type(exc).__name__}). Inspect artifacts if any were created."
            )
    if st.session_state.get("simulation_folder"):
        show_run(st.session_state["simulation_folder"])
