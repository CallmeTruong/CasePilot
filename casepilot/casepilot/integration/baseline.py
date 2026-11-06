"""Phase 0 launcher for the unmodified upstream native agent, not the CasePilot Agent."""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from casepilot.config import ProviderSettings, RunConfig
from casepilot.integration.benchmark import (
    inspect_retail,
    load_manifest,
    prepare_runtime,
    sha256_file,
)
from casepilot.models import ExperimentRun, MetricRecord
from casepilot.observability.telemetry import TraceContext, TraceSink, redact

_ENV_LOCK = threading.RLock()


@contextmanager
def provider_environment(settings: ProviderSettings, *, judge_model=None):
    """Keep credentials out of saved upstream llm_args; serialize process-global overrides."""
    values = {"OPENAI_API_KEY": settings.api_key.get_secret_value()}
    if settings.base_url:
        base = settings.base_url.get_secret_value()
        values.update(OPENAI_API_BASE=base, OPENAI_BASE_URL=base)
    with _ENV_LOCK:
        previous = {key: os.environ.get(key) for key in values}
        previous_aliases = None
        if judge_model:
            import litellm
            from tau2.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_SEMANTIC_COMMUNICATE_MODEL

            previous_aliases = dict(litellm.model_alias_map or {})
            aliases = dict(previous_aliases)
            for name in (DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_SEMANTIC_COMMUNICATE_MODEL):
                aliases[name] = judge_model
                aliases[f"openai/responses/{name}"] = judge_model
            litellm.model_alias_map = aliases
        try:
            os.environ.update(values)
            yield
        finally:
            if previous_aliases is not None:
                litellm.model_alias_map = previous_aliases
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def validate_baseline_config(config: RunConfig) -> None:
    if (
        config.architecture != "single"
        or config.tool_backend != "native"
        or config.policy_access != "full"
        or config.memory != "M0"
        or config.fault_injection
    ):
        raise ValueError("Phase 0 supports only standard Single-Agent/native/full-policy/M0")
    if not config.model or not config.user_model:
        raise ValueError("Set both model and user_model before live baseline execution")


def make_upstream_orchestrator(
    config: RunConfig, task, *, seed: int, simulation_id: str, trace=None, benchmark_revision=None
):
    """Only the simulator receives user_scenario; the agent gets public tools/policy."""
    from tau2.agent.llm_agent import LLMAgent
    from tau2.domains.retail.environment import get_environment
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.user.user_simulator import UserSimulator

    environment = get_environment()
    llm_args = {"temperature": 0, "seed": seed, "timeout": 60, "num_retries": config.llm_retries}
    agent_args, user_args = llm_args.copy(), llm_args.copy()
    for model, arguments in ((config.model, agent_args), (config.user_model, user_args)):
        if model.split("/")[-1].startswith("gpt-6"):
            # GPT-6 Chat Completions function calls require reasoning_effort=none.
            arguments.update(reasoning_effort="none", allowed_openai_params=["reasoning_effort"])
    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=config.model,
        llm_args=agent_args,
    )
    user = UserSimulator(
        llm=config.user_model,
        instructions=str(task.user_scenario),
        llm_args=user_args,
        tools=None,
    )
    return Orchestrator(
        domain="retail",
        agent=agent,
        user=user,
        environment=environment,
        task=task,
        max_steps=config.budgets.max_graph_steps,
        max_errors=3,
        seed=seed,
        simulation_id=simulation_id,
        validate_communication=True,
        timeout=600,
    )


def _save(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _collect_backend_artifacts(environment, episode_id, folder, fields):
    fields.setdefault("artifact_paths", {})
    fields.setdefault("metrics", {})
    if hasattr(environment, "journal"):
        operation_records = environment.journal.records(episode_id)
        _save(folder / f"{episode_id}.operations.json", operation_records)
        fields["artifact_paths"]["operations"] = f"{episode_id}.operations.json"
        if environment.fault is not None:
            events = environment.fault.events
            mutations = environment.mutation_events
            duplicate_count = sum(max(0, n - 1) for n in environment.mutation_commits.values())
            refreshed = False
            if events:
                target_index = next(
                    (
                        i
                        for i, op in enumerate(operation_records)
                        if op["operation_id"] == events[0]["operation_id"]
                    ),
                    -1,
                )
                refreshed = any(
                    op["tool_name"] == "get_order_details" and op["state"] == "committed"
                    for op in operation_records[target_index + 1 :]
                )
            unsafe = any(m["possibly_mutated"] for m in mutations) or duplicate_count > 0
            if events and events[0]["fault_type"] == "F3":
                unsafe = unsafe or any(
                    m["after_fault"]
                    and m["committed"]
                    and m["resource_hash"] == events[0]["resource_hash"]
                    for m in mutations
                )
            recovered = (
                bool(events)
                and not unsafe
                and getattr(environment.agent.current_state, "status", None)
                not in {"unsafe_unknown", "exhausted"}
                and (refreshed if events[0]["fault_type"] == "F3" else fields.get("reward") == 1.0)
            )
            _save(
                folder / f"{episode_id}.faults.json",
                {
                    "events": events,
                    "mutations": mutations,
                    "recovered": recovered,
                    "unsafe_recovery": unsafe,
                    "duplicate_commits": duplicate_count,
                    "fresh_read_after_fault": refreshed,
                },
            )
            fields["artifact_paths"]["faults"] = f"{episode_id}.faults.json"
            for name, value in {
                "fault_injected": bool(events),
                "recovery_success": recovered,
                "unsafe_recovery": unsafe,
                "duplicate_commits": duplicate_count,
                "recovery_steps": environment.recovery_steps,
            }.items():
                fields["metrics"][name] = MetricRecord(value=float(value), unit="count")
    for name in ("dispatch_count", "transport_errors", "recovery_steps"):
        if hasattr(environment, name):
            fields["metrics"][name] = MetricRecord(
                value=float(getattr(environment, name)), unit="count"
            )


def run_native_simulations(
    manifest_path: Path,
    config: RunConfig,
    settings: ProviderSettings,
    output: Path,
    *,
    build_orchestrator=make_upstream_orchestrator,
    validate=validate_baseline_config,
) -> Path:
    """Fresh environment per task/trial, evaluator attached, partial failures never hidden."""
    validate(config)
    manifest = load_manifest(manifest_path)
    prepare_runtime(manifest, manifest_path)
    inventory = inspect_retail(manifest_path)
    from loguru import logger
    from tau2.domains.retail.environment import get_tasks
    from tau2.runner.simulation import run_simulation
    from tau2.utils.llm_utils import set_llm_log_dir

    # Upstream debug logs may contain prompts; use the safe metadata sink below instead.
    logger.remove()
    set_llm_log_dir(None)
    tasks = {task.id: task for task in get_tasks(task_split_name=None)}
    ids = manifest.task_sets[config.split]
    if config.task_ids is not None:
        if not set(config.task_ids) <= set(ids) or len(set(config.task_ids)) != len(
            config.task_ids
        ):
            raise ValueError("Selected task IDs must be a unique subset of the selected split")
        ids = config.task_ids
    run_id = uuid.uuid4().hex
    folder = output / run_id
    folder.mkdir(parents=True, exist_ok=False)
    _save(folder / "effective-config.json", config.model_dump(mode="json"))
    _save(folder / "manifest.json", manifest.model_dump(mode="json"))
    source_root = Path(__file__).resolve().parents[1]
    _save(
        folder / "source-hashes.json",
        {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in sorted(source_root.rglob("*.py"))
        },
    )
    _save(
        folder / "schedule.json",
        [
            {"task_id": task_id, "trial": trial, "seed": config.seed + trial - 1}
            for trial in range(1, config.trials + 1)
            for task_id in ids
        ],
    )
    records = []
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    failure = False
    for trial in range(1, config.trials + 1):
        for task_id in ids:
            seed = config.seed + trial - 1
            episode_id = uuid.uuid4().hex
            record = ExperimentRun(
                experiment_id=config.experiment_id,
                config_id=config.config_id,
                config_hash=config.fingerprint(),
                run_id=run_id,
                track=config.track,
                split=config.split,
                benchmark_revision=manifest.revision,
                policy_hash=manifest.policy_hash,
                tool_hash=inventory["tool_schema_hash"],
                dependency_lock_hash=sha256_file(lock),
                model=config.model,
                user_model=config.user_model,
                inference_parameters={
                    "temperature": 0,
                    "seed": seed,
                    "judge_model": config.judge_model,
                },
                task_id=task_id,
                trial=trial,
                seed=seed,
                episode_id=episode_id,
                status="running",
            )
            trace = TraceSink(
                folder / f"{episode_id}.trace.jsonl",
                TraceContext(
                    experiment_id=config.experiment_id,
                    task_id=task_id,
                    trial=trial,
                    episode_id=episode_id,
                    run_id=run_id,
                ),
            )
            trace.emit("run_start", config_hash=config.fingerprint())
            started = time.monotonic()
            fields = record.model_dump()
            orchestrator = None
            try:
                with provider_environment(settings, judge_model=config.judge_model):
                    orchestrator = build_orchestrator(
                        config,
                        tasks[task_id],
                        seed=seed,
                        simulation_id=episode_id,
                        trace=trace,
                        benchmark_revision=manifest.revision,
                    )
                    if config.fault_injection:
                        from casepilot.evaluation.fault_evaluator import evaluate_fault_simulation

                        simulation = evaluate_fault_simulation(orchestrator)
                    else:
                        simulation = run_simulation(orchestrator)
                if simulation.reward_info is None:
                    raise RuntimeError("Upstream evaluator did not produce reward")
                # Evaluator artifacts never enter the agent's observable conversation.
                _save(
                    folder / f"{episode_id}.evaluation.json",
                    redact(simulation.reward_info.model_dump(mode="json")),
                )
                messages = [m.model_dump(mode="json") for m in simulation.messages or []]
                _save(folder / f"{episode_id}.conversation.json", redact(messages))
                fields.update(
                    status="completed",
                    reward=float(simulation.reward_info.reward),
                    termination_reason=simulation.termination_reason.value,
                    artifact_paths={
                        "conversation": f"{episode_id}.conversation.json",
                        "evaluation": f"{episode_id}.evaluation.json",
                        "trace": f"{episode_id}.trace.jsonl",
                    },
                )
                # Proxy pricing may differ from upstream estimates; keep unverified cost unknown.
                fields["metrics"] = {
                    "cost": MetricRecord(
                        value=None,
                        unit="USD",
                        undefined_reason="No verified endpoint pricing snapshot",
                    )
                }
                trace.emit("run_end", status="completed")
                state = getattr(orchestrator.agent, "current_state", None)
                if state is not None:
                    _save(
                        folder / f"{episode_id}.agent.json",
                        redact(
                            {
                                "plan": state.plan,
                                "evidence": state.evidence,
                                "status": state.status,
                                "errors": state.errors,
                                "retrieval_rounds": state.retrieval_rounds,
                            }
                        ),
                    )
                    fields["artifact_paths"]["agent"] = f"{episode_id}.agent.json"
                    for name in (
                        "input_tokens",
                        "output_tokens",
                        "llm_calls",
                        "tool_proposals",
                        "schema_valid_proposals",
                        "verification_count",
                    ):
                        known = state.usage_known or name not in {"input_tokens", "output_tokens"}
                        fields["metrics"][name] = MetricRecord(
                            value=float(getattr(state, name)) if known else None,
                            unit="count",
                            undefined_reason=None if known else "Provider usage missing",
                        )
                    fields["metrics"]["tool_calls"] = MetricRecord(
                        value=float(orchestrator.environment.dispatch_count), unit="count"
                    )
                    for role, count in state.role_calls.items():
                        fields["metrics"][role + "_calls"] = MetricRecord(
                            value=float(count), unit="count"
                        )
                    fields["metrics"]["policy_check_coverage"] = MetricRecord(
                        value=None,
                        unit="ratio",
                        undefined_reason="No independently adjudicated per-decision policy labels",
                    )
            except Exception as exc:
                import traceback

                _save(
                    folder / f"{episode_id}.error.json",
                    {
                        "error_type": type(exc).__name__,
                        "frames": [
                            {"module": Path(f.filename).name, "line": f.lineno, "function": f.name}
                            for f in traceback.extract_tb(exc.__traceback__)
                        ],
                    },
                )
                failure = True
                fields.update(status="infrastructure_error", termination_reason=type(exc).__name__)
                trace.emit("error", error_category=type(exc).__name__)
            finally:
                if orchestrator is not None:
                    _collect_backend_artifacts(orchestrator.environment, episode_id, folder, fields)
                if orchestrator is not None and hasattr(orchestrator.environment, "close"):
                    try:
                        orchestrator.environment.close()
                    except Exception:
                        failure = True
                        fields.update(
                            status="infrastructure_error", termination_reason="cleanup_error"
                        )
                        trace.emit("error", error_category="cleanup_error")
            fields["finished_at"] = datetime.now(UTC)
            fields.setdefault("metrics", {})["latency_ms"] = MetricRecord(
                value=(time.monotonic() - started) * 1000, unit="ms"
            )
            finished = ExperimentRun.model_validate(fields)
            records.append(finished.model_dump(mode="json"))
            _save(folder / "runs.json", records)
            print(f"Task {task_id}, trial {trial}: {finished.status}", flush=True)
            if failure:
                error = RuntimeError("Infrastructure failure; partial run artifacts saved")
                error.artifact_folder = str(folder)
                raise error
    return folder


def run_upstream_baseline(manifest_path, config, settings, output):
    return run_native_simulations(manifest_path, config, settings, output)
