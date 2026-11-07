"""Shared metadata vocabulary and privacy boundary."""

import math
import os
import re
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE = re.compile(
    r"api.?key|authorization|password|secret|credential|authkey|access.?token|"
    r"ground.?truth|hidden|user.?instructions|evaluation.?criteria|initial.?state|"
    r"address|email|phone|payment|customer.?id|user.?id",
    re.I,
)
EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{8,}\d(?!\w)")
KEY = re.compile(r"\b(?:sk-|sk-ant-|Bearer\s+)[A-Za-z0-9_\-]{8,}")


def redact(value: Any) -> Any:
    """Defense in depth; trace emitters still use a field allowlist, not arbitrary payloads."""
    if isinstance(value, dict):
        return {
            str(key): REDACTED if SENSITIVE.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        for key, secret in os.environ.items():
            if SENSITIVE.search(key) and len(secret) >= 8:
                value = value.replace(secret, REDACTED)
        return PHONE.sub(REDACTED, EMAIL.sub(REDACTED, KEY.sub(REDACTED, value)))
    return value


EVENTS = frozenset(
    {
        "run_start",
        "run_end",
        "agent_call",
        "agent_call_start",
        "handoff",
        "retrieval",
        "tool_call",
        "tool_dispatch",
        "tool_result",
        "verification",
        "retry",
        "fault",
        "error",
    }
)
FIELDS = frozenset(
    {
        "tool_name",
        "operation_id",
        "attempt",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "cost",
        "error_category",
        "status",
        "role",
        "config_hash",
        "retrieval_strategy",
        "evidence_ids",
        "policy_version",
        "text_hash",
        "termination_reason",
        "trace_id",
    }
)

CONTEXT_FIELDS = frozenset({"experiment_id", "task_id", "trial", "episode_id", "run_id"})


def metadata(record):
    """Copy only finite scalar metadata or string sequences; never retain business objects."""
    allowed = FIELDS | CONTEXT_FIELDS
    safe = {}
    for key, value in record.items():
        if key not in allowed:
            continue
        if isinstance(value, (str, bool, int, float)):
            if not isinstance(value, float) or math.isfinite(value):
                safe[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            safe[key] = list(value)
    return redact(safe)
