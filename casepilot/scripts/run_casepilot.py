"""Run CasePilot through the pinned Retail orchestrator and evaluator."""

import argparse
import sys
from pathlib import Path

from casepilot.config import RunConfig, load_config, load_provider_settings
from casepilot.integration.simulation_runner import run_casepilot, validate_casepilot_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env-file", default=Path(".env"), type=Path)
    parser.add_argument("--task-set", choices=["development", "final"], default="development")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--architecture", choices=["single", "multi"])
    parser.add_argument("--tool-backend", choices=["native", "mcp"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    try:
        data = load_config(args.config).model_dump()
        data.update(split=args.task_set)
        for name in ("architecture", "tool_backend", "trials"):
            if getattr(args, name) is not None:
                data[name] = getattr(args, name)
        if args.architecture or args.tool_backend:
            data["config_id"] = (
                f"casepilot_{data['architecture']}_{data['tool_backend']}_{data['policy_access']}"
            )
        if args.experiment_id:
            data["experiment_id"] = args.experiment_id
        if args.task_id:
            data["task_ids"] = args.task_id
        config = RunConfig.model_validate(data)
        validate_casepilot_config(config)
        settings = load_provider_settings(args.env_file)
    except (ValueError, OSError) as exc:
        print(
            f"Configuration preflight failed ({type(exc).__name__}). Check config and .env.",
            file=sys.stderr,
        )
        return 2
    try:
        folder = run_casepilot(args.manifest, config, settings, args.output)
    except Exception as exc:
        print(
            f"CasePilot failed ({type(exc).__name__}); inspect partial run artifacts.",
            file=sys.stderr,
        )
        return 1
    print(f"CasePilot run saved: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
