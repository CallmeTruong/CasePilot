import pytest
from pydantic import ValidationError

from casepilot.config import load_provider_settings


@pytest.fixture(autouse=True)
def clean_provider_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "BASE_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("url_key", ["OPENAI_BASE_URL", "BASE_URL"])
def test_load_private_settings_from_dotenv(tmp_path, url_key):
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY=test-private-key\n{url_key}=https://proxy.example/v1\n")
    settings = load_provider_settings(env)
    assert settings.api_key.get_secret_value() == "test-private-key"
    assert settings.base_url.get_secret_value() == "https://proxy.example/v1"
    assert "test-private-key" not in str(settings)
    assert "proxy.example" not in str(settings)


def test_environment_takes_precedence(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://file.example/v1\n")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")
    settings = load_provider_settings(env)
    assert settings.api_key.get_secret_value() == "env-key"
    assert settings.base_url.get_secret_value() == "https://env.example/v1"


def test_missing_key_has_actionable_error(tmp_path):
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        load_provider_settings(tmp_path / "missing.env")


def test_invalid_url_does_not_leak_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=private-secret\nBASE_URL=ftp://wrong.example\n")
    with pytest.raises(ValidationError) as caught:
        load_provider_settings(env)
    assert "private-secret" not in str(caught.value)
