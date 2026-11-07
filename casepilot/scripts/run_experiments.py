"""Run E1–E4 configurations in isolated worker processes."""

import argparse
import json
import os
from pathlib import Path

from casepilot.evaluation.batch import run_matrix
from casepilot.integration.provider_health import ProviderUnavailable


def main():
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--trials", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    try:
        folder = run_matrix(
            args.manifest,
            args.config,
            output=args.output,
            env_file=args.env_file,
            trials=args.trials,
            workers=args.workers,
        )
    except ProviderUnavailable as exc:
        print(f"{exc}. No benchmark jobs were started.")
        return 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"Batch preflight failed ({type(exc).__name__}); check config and .env.")
        return 2
    print(f"Batch artifacts: {folder}")
    return (
        1
        if any(
            job["status"] == "failed"
            for job in json.loads((folder / "matrix-run.json").read_text(encoding="utf-8"))["jobs"]
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
