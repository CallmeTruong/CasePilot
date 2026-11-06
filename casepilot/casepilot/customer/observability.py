"""Customer-local trace metadata; no prompts, tool arguments/results or provider secrets."""

from datetime import UTC, datetime

from casepilot.observability.telemetry import TraceSink, redact


class SessionTrace:
    def __init__(self, episode_id, rows, observer=None):
        self.episode_id, self.rows = episode_id, rows
        self.observer = observer

    def emit(self, event, **fields):
        safe = redact({key: value for key, value in fields.items() if key in TraceSink.FIELDS})
        self.rows.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "episode_id": self.episode_id,
                "event": event,
                **safe,
            }
        )
        if self.observer:
            self.observer(event, fields)
