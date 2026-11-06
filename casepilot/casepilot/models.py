"""Provenance and evaluator-side records; these are not Agent state or prompt objects."""

from datetime import UTC, datetime
from typing import Any, Literal, Self

from pydantic import AwareDatetime, Field, field_validator, model_validator

from casepilot.config import NonEmpty, StrictModel


class MetricRecord(StrictModel):
    value: float | None
    unit: NonEmpty
    numerator: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    denominator: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    undefined_reason: NonEmpty | None = None

    @model_validator(mode="after")
    def check_defined(self) -> Self:
        import math

        if self.value is None and not self.undefined_reason:
            raise ValueError("Undefined metrics require undefined_reason")
        if self.value is not None and (not math.isfinite(self.value) or self.undefined_reason):
            raise ValueError("Defined metrics must be finite and have no undefined_reason")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("numerator and denominator must be provided together")
        if self.denominator == 0 and self.value is not None:
            raise ValueError("Zero denominator requires an undefined metric")
        return self


class ExperimentRun(StrictModel):
    experiment_id: NonEmpty
    config_id: NonEmpty
    config_hash: NonEmpty
    run_id: NonEmpty
    track: Literal["standard", "custom"] = "standard"
    split: Literal["development", "final"] = "development"
    benchmark_revision: NonEmpty
    policy_hash: NonEmpty
    tool_hash: NonEmpty
    dependency_lock_hash: NonEmpty
    model: NonEmpty
    user_model: NonEmpty
    inference_parameters: dict[str, Any] = Field(default_factory=dict)
    task_id: NonEmpty
    trial: int = Field(ge=1)
    seed: int = Field(ge=0)
    episode_id: NonEmpty
    started_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: AwareDatetime | None = None
    status: Literal["scheduled", "running", "completed", "infrastructure_error"] = "scheduled"
    termination_reason: NonEmpty | None = None
    reward: float | None = Field(default=None, allow_inf_nan=False)
    metrics: dict[str, MetricRecord] = Field(default_factory=dict)
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    @field_validator("started_at", "finished_at")
    @classmethod
    def utc_timestamps(cls, value):
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def check_lifecycle(self) -> Self:
        if self.finished_at and self.finished_at < self.started_at:
            raise ValueError("finished_at precedes started_at")
        if self.status == "completed" and (self.finished_at is None or self.reward is None):
            raise ValueError("Completed runs require finished_at and evaluator reward")
        return self

    def agent_context(self) -> dict[str, str]:
        """Explicit allowlist prevents evaluator provenance or reward entering model context."""
        return {"episode_id": self.episode_id}
