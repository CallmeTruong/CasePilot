"""Synthetic multi-session experiments; verified runtime facts, never standard task history."""

import json
import time
import uuid
from pathlib import Path

from casepilot.config import RunConfig, load_provider_settings
from casepilot.integration.baseline import _save, provider_environment
from casepilot.integration.benchmark import load_manifest, prepare_runtime, sha256_file
from casepilot.memory.episodic import HistoryStore
from casepilot.observability.telemetry import TraceContext, TraceSink, redact


def _session(
    env,
    config,
    *,
    question,
    case_reference,
    memory_scope,
    store,
    run_id,
    folder,
    settings,
    trial,
    benchmark_revision,
):
    from tau2.data_model.message import UserMessage

    from casepilot.integration.tau2_agent import CasePilotAgent
    from casepilot.integration.tool_adapter import GuardedEnvironment
    from casepilot.memory.checkpoint import CheckpointManager

    episode = uuid.uuid4().hex
    trace = TraceSink(
        folder / f"{episode}.trace.jsonl",
        TraceContext(
            experiment_id="E4", task_id=memory_scope, trial=trial, episode_id=episode, run_id=run_id
        ),
    )
    checkpoint = (
        CheckpointManager(
            folder / "checkpoints.sqlite3",
            benchmark_revision=benchmark_revision,
            config_hash=config.fingerprint(),
        )
        if config.checkpointing
        else None
    )
    agent = CasePilotAgent(
        env.get_tools(),
        env.get_policy(),
        config,
        episode_id=episode,
        memory_store=store,
        memory_scope=memory_scope,
        case_reference=case_reference,
        trace=trace,
        checkpoint_manager=checkpoint,
    )
    wrapper = GuardedEnvironment(env, agent, journal_path=folder / "operations.sqlite")
    state = agent.get_init_state()
    incoming = UserMessage(role="user", content=question)
    started = time.perf_counter()
    try:
        with provider_environment(settings):
            for _ in range(config.budgets.max_graph_steps):
                reply, state = agent.generate_next_message(incoming, state)
                if not reply.tool_calls:
                    return {
                        "episode_id": episode,
                        "response": reply.content or "",
                        "state": state,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "journal": wrapper.journal,
                        "wrapper": wrapper,
                    }
                incoming = wrapper.get_response(reply.tool_calls[0])
        raise RuntimeError("Memory session exceeded step budget")
    except Exception:
        wrapper.close()
        raise


def run_memory_scenario(manifest_path, base_values, scenario, *, trial, output, env_file=".env"):
    """One verified source session is shared by paired M0/M1 follow-up sessions."""
    manifest = load_manifest(manifest_path)
    prepare_runtime(manifest, manifest_path)
    from loguru import logger
    from tau2.domains.retail.environment import get_environment
    from tau2.utils.llm_utils import set_llm_log_dir

    logger.remove()
    set_llm_log_dir(None)
    settings = load_provider_settings(env_file)
    run_id = uuid.uuid4().hex
    folder = Path(output) / run_id
    folder.mkdir(parents=True, exist_ok=False)
    scope = run_id + ":" + scenario["scenario_id"]
    env = get_environment()
    customer = env.tools.db.users[scenario["customer_id"]]
    old_zip = customer.address.zip
    reference = (
        "CASE_"
        + uuid.uuid5(uuid.NAMESPACE_URL, f"{scenario['scenario_id']}:{trial}:source").hex[:16]
    )
    source_config = RunConfig.model_validate(base_values | {"memory": "M0", "track": "custom"})
    lock_hash = sha256_file(Path(__file__).resolve().parents[2] / "uv.lock")
    _save(folder / "manifest.json", manifest.model_dump(mode="json"))
    _save(
        folder / "effective-configs.json",
        {
            mode: RunConfig.model_validate(
                base_values | {"memory": mode, "track": "custom"}
            ).model_dump(mode="json")
            for mode in ("M0", "M1")
        },
    )
    _save(
        folder / "schedule.json",
        {
            "scenario_id": scenario["scenario_id"],
            "trial": trial,
            "seed": source_config.seed,
            "modes": ["M0", "M1"],
            "dependency_lock_hash": lock_hash,
        },
    )
    store = HistoryStore(folder / "history.sqlite")
    try:
        source = _session(
            env,
            source_config,
            question=(
                f"My email is {customer.email}. Please verify my identity and look up my profile."
            ),
            case_reference=reference,
            memory_scope=scope,
            store=store,
            run_id=run_id,
            folder=folder,
            settings=settings,
            trial=trial,
            benchmark_revision=manifest.revision,
        )
    except Exception as exc:
        store.close()
        exc.artifact_folder = str(folder)
        raise
    rows = []
    try:
        history = store.record_verified_case(
            scope=scope, state=source["state"], journal=source["journal"]
        )
        if scenario["stale_change"]:
            customer.address.zip = scenario["new_zip"]
        for mode in ("M0", "M1"):
            config = RunConfig.model_validate(base_values | {"memory": mode, "track": "custom"})
            question = f"My email is {customer.email}. Please verify my identity, then tell me the "
            question += "case reference assigned to my profile lookup in the previous session. "
            if scenario["stale_change"]:
                question += "Also look up and tell me my current ZIP code."
            current_ref = (
                "CASE_"
                + uuid.uuid5(uuid.NAMESPACE_URL, f"{scenario['scenario_id']}:{trial}:followup").hex[
                    :16
                ]
            )
            session = _session(
                env,
                config,
                question=question,
                case_reference=current_ref,
                memory_scope=scope,
                store=store,
                run_id=run_id,
                folder=folder,
                settings=settings,
                trial=trial,
                benchmark_revision=manifest.revision,
            )
            try:
                response, state = session["response"], session["state"]
                fresh = not scenario["stale_change"] or (
                    scenario["new_zip"] in response
                    and any(
                        t["name"] == "get_user_details" and not t["error"]
                        for t in state.tool_results
                    )
                )
                retrieved = bool(state.memory_history_ids)
                stale_error = bool(scenario["stale_change"] and old_zip in response and not fresh)
                row = {
                    "experiment_id": "E4",
                    "track": "custom",
                    "status": "completed",
                    "run_id": run_id,
                    "scenario_id": scenario["scenario_id"],
                    "trial": trial,
                    "memory": mode,
                    "benchmark_revision": manifest.revision,
                    "policy_hash": manifest.policy_hash,
                    "dependency_lock_hash": lock_hash,
                    "config_hash": config.fingerprint(),
                    "cross_session_success": reference in response
                    and fresh
                    and state.customer_id == customer.user_id,
                    "memory_retrieval_accuracy": float(
                        history.history_id in state.memory_history_ids
                    )
                    / len(state.memory_history_ids)
                    if retrieved
                    else None,
                    "retrieved_records": len(state.memory_history_ids),
                    "memory_retrieval_undefined_reason": None
                    if retrieved
                    else "no retrieved records",
                    "metric_phase": "followup_only",
                    "source_latency_ms": source["latency_ms"],
                    "source_input_tokens": source["state"].input_tokens,
                    "source_output_tokens": source["state"].output_tokens,
                    "stale_memory_error": stale_error,
                    "latency_ms": session["latency_ms"],
                    "input_tokens": state.input_tokens if state.usage_known else None,
                    "output_tokens": state.output_tokens if state.usage_known else None,
                    "model": config.model,
                    "cost": None,
                    "source_episode_id": source["episode_id"],
                    "episode_id": session["episode_id"],
                }
                rows.append(row)
                (folder / "memory-results.json").write_text(
                    json.dumps(rows, indent=2), encoding="utf-8"
                )
                transcript = [m.model_dump(mode="json") for m in state.messages]
                (folder / f"{session['episode_id']}.conversation.json").write_text(
                    json.dumps(redact(transcript), ensure_ascii=False, indent=2), encoding="utf-8"
                )
            finally:
                session["wrapper"].close()
        (folder / "memory-results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    except Exception as exc:
        exc.artifact_folder = str(folder)
        raise
    finally:
        source["wrapper"].close()
        store.close()
    return str(folder)
