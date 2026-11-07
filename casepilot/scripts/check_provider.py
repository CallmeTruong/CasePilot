"""Check model visibility and real inference using .env, without logging private settings."""

import argparse
import json
from pathlib import Path

from casepilot.config import load_config, load_provider_settings
from casepilot.integration.provider_health import probe_provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/provider-health.json"))
    args = parser.parse_args()
    try:
        settings = load_provider_settings(args.env_file)
        config = load_config(args.config)
        checks = [(config.model, "models"), (config.model, "chat")]
        if config.user_model != config.model:
            checks.append((config.user_model, "chat"))
        if config.judge_model:
            checks.append(
                (config.judge_model, "responses" if "/responses/" in config.judge_model else "chat")
            )
        results = [probe_provider(settings, model=model, api=api) for model, api in checks]
    except (ValueError, OSError) as exc:
        print(f"Provider configuration failed ({type(exc).__name__}); check config and .env.")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(
            f"{result['api']}: HTTP {result['http_status']} · operational={result['operational']}"
            f" · error={result['error_type']}"
        )
    return 0 if all(r["operational"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
