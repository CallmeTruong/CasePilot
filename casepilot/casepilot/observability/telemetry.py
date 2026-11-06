"""Metadata-only JSONL traces. Observable conversation artifacts need separate access controls."""

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, StringConstraints

from casepilot.config import StrictModel

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


TraceID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]+$", min_length=1)]


class TraceContext(StrictModel):
    experiment_id: TraceID
    task_id: TraceID
    trial: int = Field(ge=1)
    episode_id: TraceID
    run_id: TraceID


class TraceSink:
    EVENTS = frozenset(
        {
            "run_start",
            "run_end",
            "agent_call",
            "agent_call_start",
            "handoff",
            "retrieval",
            "tool_call",
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

    def __init__(self, path: Path | str, context: TraceContext):
        self.path = Path(path)
        self.context = context.model_copy(deep=True)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        if event not in self.EVENTS:
            raise ValueError("Unsupported trace event")
        record = {
            **self.context.model_dump(),
            "event": event,
            **redact({key: value for key, value in fields.items() if key in self.FIELDS}),
        }
        # Generated timestamps are trusted metadata, not customer phone numbers.
        record["timestamp"] = datetime.now(UTC).isoformat()
        from casepilot.observability.tracing import export_event

        trace_id = export_event(event, record)
        if trace_id:
            record["trace_id"] = trace_id
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
