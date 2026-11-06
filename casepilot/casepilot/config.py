"""Validated experiment configuration; research tracks are never selected implicitly."""

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, model_validator

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)


class Budgets(StrictModel):
    max_graph_steps: int = Field(default=40, ge=1)
    max_replans: int = Field(default=2, ge=0)
    max_tokens: int = Field(default=32000, ge=1)
    max_retrieval_rounds: int = Field(default=2, ge=1, le=2)


class MCPConfig(StrictModel):
    transport: Literal["stdio"] = "stdio"
    timeout_seconds: float = Field(default=15.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)


class RunConfig(StrictModel):
    experiment_id: NonEmpty = "phase0"
    config_id: NonEmpty = "native_baseline"
    track: Literal["standard", "custom"] = "standard"
    split: Literal["development", "final"] = "development"
    architecture: Literal["single", "multi"] = "single"
    tool_backend: Literal["native", "mcp"] = "native"
    policy_access: Literal["full", "bm25", "dense", "hybrid", "hybrid_rerank"] = "full"
    memory: Literal["M0", "M1", "M2"] = "M0"
    fault_injection: bool = False
    recovery_strategy: Literal["S0", "S1", "S2"] = "S2"
    fault_type: Literal["F1", "F2", "F3"] | None = None
    fault_target: NonEmpty | None = None
    fault_trigger_attempt: int = Field(default=1, ge=1)
    model: NonEmpty | None = None
    user_model: NonEmpty | None = None
    judge_model: NonEmpty | None = None
    seed: int = Field(default=42, ge=0)
    trials: int = Field(default=1, ge=1)
    task_ids: list[NonEmpty] | None = Field(default=None, min_length=1)
    scenario_id: NonEmpty | None = None
    llm_retries: int = Field(default=2, ge=0, le=3)
    checkpointing: bool = True
    budgets: Budgets = Field(default_factory=Budgets)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    @model_validator(mode="after")
    def check_track(self) -> Self:
        if self.track == "standard" and (
            self.policy_access != "full"
            or self.memory != "M0"
            or self.fault_injection
            or self.judge_model is not None
        ):
            raise ValueError("RAG, faults and episodic memory require the explicit custom track")
        return self

    def fingerprint(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def load_config(path: Path | str) -> RunConfig:
    """Read YAML without executing tags. Model environment defaults never override explicit IDs."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for field, env_name in (("model", "CASEPILOT_MODEL"), ("user_model", "CASEPILOT_USER_MODEL")):
        if not data.get(field) and os.environ.get(env_name, "").strip():
            data[field] = os.environ[env_name].strip()
    return RunConfig.model_validate(data)


class ProviderSettings(StrictModel):
    """Private runtime settings; never put this object in artifacts or trace context."""

    api_key: SecretStr
    base_url: SecretStr | None = None

    @model_validator(mode="after")
    def check_settings(self) -> Self:
        if not self.api_key.get_secret_value().strip():
            raise ValueError("Set OPENAI_API_KEY in the environment or local .env")
        if self.base_url is not None:
            parsed = urlsplit(self.base_url.get_secret_value())
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("OPENAI_BASE_URL/BASE_URL must be an HTTP(S) base URL")
        return self


def load_provider_settings(env_file: Path | str = ".env") -> ProviderSettings:
    """Environment wins over .env; explicit loading has no process-wide side effects."""
    values = {**dotenv_values(env_file), **os.environ}
    key = (values.get("OPENAI_API_KEY") or "").strip()
    url = (values.get("OPENAI_BASE_URL") or values.get("BASE_URL") or "").strip()
    return ProviderSettings(api_key=SecretStr(key), base_url=SecretStr(url) if url else None)
