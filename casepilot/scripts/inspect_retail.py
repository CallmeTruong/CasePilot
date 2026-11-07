"""Inspect pinned Retail public APIs and tool schemas without making LLM calls."""

import argparse
import json
import sys
from pathlib import Path

from casepilot.integration.benchmark import inspect_retail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/inspection.json"))
    args = parser.parse_args()
    try:
        inventory = inspect_retail(args.manifest)
    except (ValueError, OSError) as exc:
        print(f"Retail preflight failed: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(inventory['tool_schemas'])} Retail tools at {inventory['revision']}")
    print(f"Public API inventory: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
