"""Run the pinned upstream native Retail baseline (Phase 0; requires provider credentials)."""

import argparse
import sys
from pathlib import Path

from casepilot.config import RunConfig, load_config, load_provider_settings
from casepilot.integration.baseline import run_upstream_baseline, validate_baseline_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env-file", default=Path(".env"), type=Path)
    parser.add_argument("--task-set", choices=["development", "final"], default="development")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        data = load_config(args.config).model_dump()
        data["split"] = args.task_set
        if args.trials is not None:
            data["trials"] = args.trials
        config = RunConfig.model_validate(data)
        validate_baseline_config(config)
        settings = load_provider_settings(args.env_file)
    except (ValueError, OSError):
        print(
            "Invalid baseline config or missing OPENAI_API_KEY. Check --config/--env-file; "
            "set OPENAI_BASE_URL (or BASE_URL) for your endpoint.",
            file=sys.stderr,
        )
        return 2
    try:
        folder = run_upstream_baseline(args.manifest, config, settings, args.output)
    except (ValueError, OSError) as exc:
        print(
            f"Baseline preflight failed: {type(exc).__name__}. Run inspect_retail.py first.",
            file=sys.stderr,
        )
        return 2
    except Exception:
        print(
            "Baseline infrastructure/provider failure; inspect partial run metadata. "
            "Provider error payloads are omitted to protect credentials.",
            file=sys.stderr,
        )
        return 1
    print(f"Upstream baseline saved: {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
