"""Explicit metadata-only export using a reusable client owned by the dispatcher worker."""

import os
import threading
from datetime import UTC, datetime
from uuid import uuid4

from langsmith import Client

from casepilot.observability.events import EVENTS, metadata

_client = None
_lock = threading.Lock()


def enabled():
    """Only the application flag in process environment authorizes cloud export."""
    return os.environ.get("CASEPILOT_LANGSMITH_TRACING", "").strip().lower() in {"true", "1", "yes"}


def export_event(event, record):
    """Synchronous worker entry point; application callers use Dispatcher.dispatch."""
    global _client
    if not enabled() or event not in EVENTS:
        return None
    safe = metadata(record)
    with _lock:
        if _client is None:
            _client = Client(
                hide_inputs=True,
                hide_outputs=True,
                omit_traced_runtime_info=True,
                auto_batch_tracing=False,
            )
        identifier = uuid4()
        now = datetime.now(UTC)
        _client.create_run(
            id=identifier,
            name=event,
            run_type="chain",
            inputs={},
            outputs={},
            start_time=now,
            end_time=now,
            extra={"metadata": safe},
        )
    return str(identifier)


def close():
    global _client
    with _lock:
        client, _client = _client, None
        if client is not None:
            client.close()
