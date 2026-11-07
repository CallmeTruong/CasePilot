import uuid
from types import SimpleNamespace

import pytest

from casepilot.integration.tau2.tools import describe_tools

pytestmark = pytest.mark.integration


def test_each_mcp_bridge_owns_a_distinct_environment_and_identity(retail_environment):
    from tau2.domains.retail.environment import get_environment

    from casepilot.agents.state import CaseState
    from casepilot.agents.verifier import PolicyGate
    from casepilot.integration.tau2.dispatcher import GuardedEnvironment

    wrappers = []
    try:
        for env in (retail_environment, get_environment()):
            episode_id = uuid.uuid4().hex
            agent = SimpleNamespace(
                episode_id=episode_id,
                current_state=CaseState(episode_id=episode_id),
                engine=SimpleNamespace(gate=PolicyGate(describe_tools(env.get_tools()))),
            )
            wrappers.append(GuardedEnvironment(env, agent, backend="mcp"))
        first, second = wrappers
        assert first.tools is not second.tools
        assert first.bridge.address != second.bridge.address
        assert first.bridge.authkey != second.bridge.authkey
        response = first.bridge.request(
            {"episode_id": second.agent.episode_id, "action": "list_tools", "kind": "read"}
        )
        assert response["protocol_error"] == "episode_mismatch"
        assert first.dispatch_count == second.dispatch_count == 0
    finally:
        for wrapper in wrappers:
            wrapper.close()
