import os

import pytest
from pydantic import SecretStr

from casepilot.config import ProviderSettings, RunConfig
from casepilot.integration.baseline import provider_environment, validate_baseline_config


@pytest.mark.parametrize(
    "values",
    [
        {"architecture": "multi"},
        {"tool_backend": "mcp"},
        {"track": "custom", "policy_access": "bm25"},
        {"track": "custom", "memory": "M1"},
        {"track": "custom", "fault_injection": True},
        {"model": None},
    ],
)
def test_upstream_baseline_never_silently_ignores_requested_features(values):
    config = RunConfig.model_validate({"model": "gpt-4o", "user_model": "gpt-4o"} | values)
    with pytest.raises(ValueError):
        validate_baseline_config(config)


def test_provider_key_and_proxy_restored_after_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "old-key")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = ProviderSettings(
        api_key=SecretStr("new-key"), base_url=SecretStr("https://proxy.example/v1")
    )
    with pytest.raises(RuntimeError), provider_environment(settings):
        assert os.environ["OPENAI_API_KEY"] == "new-key"
        assert os.environ["OPENAI_API_BASE"] == "https://proxy.example/v1"
        raise RuntimeError("simulated error")
    assert os.environ["OPENAI_API_KEY"] == "old-key"
    assert "OPENAI_API_BASE" not in os.environ
    assert "OPENAI_BASE_URL" not in os.environ
