"""Explicit opt-in local Giskard red team; calls the configured model endpoint."""

import argparse
import asyncio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="Allow paid provider calls on synthetic data"
    )
    parser.add_argument("--mode", choices=["curated", "scan", "replay"], default="scan")
    parser.add_argument("--profile", choices=["smoke", "focused", "full"], default="focused")
    parser.add_argument("--max-scenarios", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-model-calls", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/security"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Provider calls require --live; use pytest tests/security for offline checks")
    from casepilot.config import load_config, load_provider_settings
    from casepilot.security.redteam import RedteamConfig, run_redteam

    root = Path(__file__).resolve().parents[1]
    try:
        config = RedteamConfig(
            mode=args.mode,
            profile=args.profile,
            max_scenarios=args.max_scenarios,
            max_turns=args.max_turns,
            max_model_calls=args.max_model_calls,
            timeout_seconds=args.timeout_seconds,
            seed=args.seed,
        )
        settings = load_provider_settings(root / ".env")
        model = args.model or load_config(root / "configs/default.yaml").model
        folder, result = asyncio.run(
            run_redteam(
                config, settings, model=model, output_root=args.output, replay_path=args.suite
            )
        )
    except Exception:
        print(
            "Red-team setup failed. Check local configuration and optional security dependencies."
        )
        return 2
    counts = {key: result.get(key) for key in ("passed", "failed", "errored", "skipped")}
    print(f"Red-team results: {folder}; completed={result['completed']}; {counts}")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
