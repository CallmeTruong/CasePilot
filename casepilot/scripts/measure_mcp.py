"""Measure five real local read calls per backend, without provider requests."""

import json
import platform
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from casepilot.agents.state import CaseState
from casepilot.agents.verifier import PolicyGate
from casepilot.integration.benchmark import load_manifest, prepare_runtime


def main():
    path = Path("configs/benchmark-manifest.json")
    manifest = load_manifest(path)
    prepare_runtime(manifest, path)
    from loguru import logger
    from tau2.data_model.message import ToolCall
    from tau2.domains.retail.environment import get_environment

    from casepilot.integration.tau2.dispatcher import GuardedEnvironment

    logger.remove()
    rows = []
    for backend in ("native", "mcp"):
        env = get_environment()
        customer = next(iter(env.tools.db.users.values()))
        episode = uuid.uuid4().hex
        agent = SimpleNamespace(
            episode_id=episode,
            current_state=CaseState(episode_id=episode, customer_id=customer.user_id),
            engine=SimpleNamespace(gate=PolicyGate(env.get_tools())),
        )
        wrapper = GuardedEnvironment(env, agent, backend=backend)
        try:
            for iteration in range(1, 6):
                started = time.perf_counter()
                response = wrapper.get_response(
                    ToolCall(
                        id=uuid.uuid4().hex,
                        name="get_user_details",
                        arguments={"user_id": customer.user_id},
                    )
                )
                rows.append(
                    {
                        "backend": backend,
                        "iteration": iteration,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        "error": response.error,
                    }
                )
                assert not response.error
            assert wrapper.dispatch_count == 5 and wrapper.transport_errors == 0
        finally:
            wrapper.close()
    root = Path("results/parity")
    root.mkdir(parents=True, exist_ok=True)
    (root / "measurements.json").write_text(
        json.dumps(
            {
                "revision": manifest.revision,
                "platform": platform.platform(),
                "protocol": "five reads/backend; process per MCP call",
                "observations": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Measurements saved: results/parity/measurements.json")


if __name__ == "__main__":
    main()
