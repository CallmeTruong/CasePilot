"""Customer-local trace metadata; no prompts, tool arguments/results or provider secrets."""

from datetime import UTC, datetime

from casepilot.observability.events import EVENTS, FIELDS, metadata


class SessionTrace:
    def __init__(self, episode_id, rows, observer=None, *, dispatcher=None):
        from casepilot.observability.dispatcher import get_dispatcher

        self.episode_id, self.rows = episode_id, rows
        self.observer = observer
        self.dispatcher = dispatcher if dispatcher is not None else get_dispatcher()

    def emit(self, event, **fields):
        if event not in EVENTS:
            raise ValueError("Unsupported trace event")
        safe = metadata(
            {
                "episode_id": self.episode_id,
                **{key: value for key, value in fields.items() if key in FIELDS},
            }
        )
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event, **safe}
        trace_id = self.dispatcher.dispatch(event, safe)
        if trace_id:
            record["trace_id"] = trace_id
        self.rows.append(record)
        if self.observer:
            self.observer(event, fields)
