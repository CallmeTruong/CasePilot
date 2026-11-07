import pytest
from pydantic import ValidationError

from casepilot.config import RunConfig, load_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("policy_access", "hybrid"),
        ("fault_injection", True),
        ("memory", "M1"),
    ],
)
def test_standard_rejects_custom_research(field, value):
    with pytest.raises(ValidationError, match="custom"):
        RunConfig(**{field: value})


def test_custom_allows_research():
    config = RunConfig(track="custom", policy_access="hybrid", memory="M1")
    assert config.track == "custom"


@pytest.mark.parametrize(
    "values",
    [
        {"trials": 0},
        {"budgets": {"max_tokens": 0}},
        {"budgets": {"max_retrieval_rounds": 3}},
        {"mcp": {"timeout_seconds": 0}},
        {"mcp": {"max_retries": -1}},
        {"unknown": "typo"},
        {"fault_injection": "false"},
    ],
)
def test_invalid_configuration_fails(values):
    with pytest.raises(ValidationError):
        RunConfig(**values)


def test_env_models_and_stable_hash(tmp_path, monkeypatch):
    path = tmp_path / "run.yaml"
    path.write_text("track: standard\n", encoding="utf-8")
    monkeypatch.setenv("CASEPILOT_MODEL", "provider/model")
    monkeypatch.setenv("CASEPILOT_USER_MODEL", "provider/user")
    config = load_config(path)
    assert config.model == "provider/model"
    assert config.user_model == "provider/user"
    assert config.fingerprint() == load_config(path).fingerprint()


def test_explicit_models_take_priority(tmp_path, monkeypatch):
    path = tmp_path / "run.yaml"
    path.write_text("model: configured/model\n", encoding="utf-8")
    monkeypatch.setenv("CASEPILOT_MODEL", "env/model")
    assert load_config(path).model == "configured/model"


def test_non_mapping_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("- not-a-config\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)
