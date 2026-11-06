"""Construct CasePilot without passing task/evaluator data to the Agent."""

import hashlib
from pathlib import Path

from casepilot.config import RunConfig
from casepilot.integration.baseline import make_upstream_orchestrator, run_native_simulations
from casepilot.integration.environment_adapter import PublicRetailContext


def validate_casepilot_config(config: RunConfig):
    if config.memory != "M0":
        raise ValueError("Memory mode is not implemented yet")
    if config.fault_injection and (not config.fault_type or not config.fault_target):
        raise ValueError("Fault injection requires fault_type and actual toolkit fault_target")
    if not config.model or not config.user_model:
        raise ValueError("Model identifiers are required")


def make_casepilot_orchestrator(
    config, task, *, seed, simulation_id, trace=None, benchmark_revision=None
):
    from casepilot.integration.tau2_agent import CasePilotAgent
    from casepilot.integration.tool_adapter import GuardedEnvironment

    orchestrator = make_upstream_orchestrator(config, task, seed=seed, simulation_id=simulation_id)
    context = PublicRetailContext.from_environment(orchestrator.environment)
    seeded = RunConfig.model_validate(config.model_dump() | {"seed": seed})
    if config.fault_injection and config.fault_target not in {tool.name for tool in context.tools}:
        raise ValueError("Fault target is not in the pinned toolkit")
    checkpoint = None
    if config.checkpointing:
        from casepilot.memory.checkpoint import CheckpointManager

        checkpoint = CheckpointManager(
            Path(".cache/checkpoints")
            / (hashlib.sha256(simulation_id.encode()).hexdigest() + ".sqlite3"),
            benchmark_revision=benchmark_revision or "unrecorded",
            config_hash=seeded.fingerprint(),
        )
    try:
        agent = CasePilotAgent(
            context.tools,
            context.policy,
            seeded,
            episode_id=simulation_id,
            trace=trace,
            checkpoint_manager=checkpoint,
        )
    except Exception:
        if checkpoint:
            checkpoint.close()
        raise
    orchestrator.agent = agent
    fault = None
    if config.fault_injection:
        from casepilot.reliability.fault_injector import FaultInjector

        fault = FaultInjector(
            config.fault_type,
            config.fault_target,
            seed=seed,
            trigger_attempt=config.fault_trigger_attempt,
        )
    orchestrator.environment = GuardedEnvironment(
        orchestrator.environment,
        agent,
        backend=config.tool_backend,
        timeout_seconds=config.mcp.timeout_seconds,
        recovery=config.recovery_strategy,
        max_retries=config.mcp.max_retries,
        backoff_seconds=config.mcp.retry_backoff_seconds,
        fault=fault,
    )
    return orchestrator


def run_casepilot(manifest_path, config, settings, output):
    return run_native_simulations(
        manifest_path,
        config,
        settings,
        output,
        build_orchestrator=make_casepilot_orchestrator,
        validate=validate_casepilot_config,
    )
