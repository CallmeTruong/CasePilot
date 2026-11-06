import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from casepilot.config import NonEmpty, StrictModel


def fingerprint(
    tool: str, arguments: dict, customer: str | None, revision: int, episode_id: str | None = None
) -> str:
    return hashlib.sha256(
        json.dumps(
            [episode_id, tool, arguments, customer, revision], sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


class Confirmation(StrictModel):
    episode_id: NonEmpty
    fingerprint: NonEmpty
    tool_name: NonEmpty
    arguments: dict[str, Any]
    plan_revision: int = Field(ge=0)
    status: Literal["pending", "confirmed", "invalidated", "consumed"] = "pending"
    user_turn: int = Field(ge=0)
    user_message_id: str | None = None
    confirmed_at: datetime | None = None


class CaseState(StrictModel):
    episode_id: NonEmpty
    messages: list[Any] = Field(default_factory=list)
    customer_id: str | None = None
    owned_order_ids: set[str] = Field(default_factory=set)
    required_refresh_order_ids: set[str] = Field(default_factory=set)
    case_id: NonEmpty | None = None
    intent: str | None = None
    plan: dict | None = None
    evidence: list[dict] = Field(default_factory=list)
    memory_history_ids: list[str] = Field(default_factory=list)
    pending_tool_calls: dict[str, dict] = Field(default_factory=dict)
    tool_results: list[dict] = Field(default_factory=list)
    confirmation: Confirmation | None = None
    plan_revision: int = Field(default=0, ge=0)
    user_turn: int = Field(default=0, ge=0)
    completed_actions: dict[str, int] = Field(default_factory=dict)
    errors: list[dict] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    retrieval_rounds: int = Field(default=0, ge=0, le=2)
    graph_steps: int = Field(default=0, ge=0)
    token_usage: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    usage_known: bool = True
    llm_calls: int = Field(default=0, ge=0)
    role_calls: dict[str, int] = Field(default_factory=dict)
    tool_proposals: int = Field(default=0, ge=0)
    schema_valid_proposals: int = Field(default=0, ge=0)
    verification_count: int = Field(default=0, ge=0)
    status: Literal[
        "active",
        "waiting_info",
        "waiting_confirmation",
        "completed",
        "rejected",
        "cancelled",
        "exhausted",
        "unsafe_unknown",
    ] = "active"
    termination_reason: str | None = None
