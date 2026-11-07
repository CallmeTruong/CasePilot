"""Metadata-only JSONL traces. Observable conversation artifacts need separate access controls."""

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, StringConstraints

from casepilot.config import StrictModel
from casepilot.observability.events import EVENTS, FIELDS, metadata

TraceID = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]+$", min_length=1)]


class TraceContext(StrictModel):
    experiment_id: TraceID
    task_id: TraceID
    trial: int = Field(ge=1)
    episode_id: TraceID
    run_id: TraceID


class TraceSink:
    EVENTS = EVENTS
    FIELDS = FIELDS

    def __init__(self, path: Path | str, context: TraceContext, *, dispatcher=None):
        from casepilot.observability.dispatcher import get_dispatcher

        self.dispatcher = dispatcher if dispatcher is not None else get_dispatcher()
        self.path = Path(path)
        self.context = context.model_copy(deep=True)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        if event not in self.EVENTS:
            raise ValueError("Unsupported trace event")
        record = {
            **metadata(self.context.model_dump()),
            "event": event,
            **metadata({key: value for key, value in fields.items() if key in self.FIELDS}),
        }
        # Generated timestamps are trusted metadata, not customer phone numbers.
        record["timestamp"] = datetime.now(UTC).isoformat()
        trace_id = self.dispatcher.dispatch(event, record)
        if trace_id:
            record["trace_id"] = trace_id
        line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
