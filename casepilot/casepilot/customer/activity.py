"""Bounded business-data previews and transient live workspace activity."""

import json
import os
import re
import time
from copy import deepcopy

PRIVATE = re.compile(
    r"api.?key|authorization|password|secret|credential|authkey|access.?token|"
    r"ground.?truth|hidden|instructions|evaluation|prompt|base.?url|endpoint",
    re.I,
)
KEY = re.compile(r"\b(?:sk-|Bearer\s+)[A-Za-z0-9_\-]{8,}")


def preview(value, secrets=()):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
    known = [*secrets, *(v for k, v in os.environ.items() if PRIVATE.search(k) and len(v) >= 8)]
    budget = [10000]

    def visit(item, depth=0):
        if depth > 6 or budget[0] <= 0:
            return "[truncated]"
        if isinstance(item, dict):
            result = {}
            for key, child in list(item.items())[:30]:
                safe_key = clean(str(key))
                result[safe_key] = (
                    "[REDACTED]" if PRIVATE.search(str(key)) else visit(child, depth + 1)
                )
            if len(item) > 30:
                result["…"] = "[truncated]"
            return result
        if isinstance(item, (list, tuple)):
            result = [visit(child, depth + 1) for child in item[:20]]
            return result + (["[truncated]"] if len(item) > 20 else [])
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return clean(str(item))

    def clean(text):
        for secret in known:
            if secret:
                text = text.replace(secret, "[REDACTED]")
        text = KEY.sub("[REDACTED]", text)
        limit = min(1500, max(budget[0], 0))
        budget[0] -= min(len(text), limit)
        return text[:limit] + (" [truncated]" if len(text) > limit else "")

    return visit(value)


class ActivityRecorder:
    def __init__(self, rows, secrets=()):
        self.rows, self.secrets = rows, secrets
        self.turn = 0
        self.callback = None
        self.active = False
        self.starts = {}

    def begin(self, callback):
        self.turn += 1
        self.callback, self.active = callback, True

    def notify(self):
        if self.callback:
            try:
                self.callback(deepcopy(self.rows))
            except Exception:
                # A disconnected view cannot interrupt a committed business operation.
                self.callback = None

    def start(self, kind, name, **fields):
        row = {
            "id": len(self.rows) + 1,
            "turn": self.turn,
            "kind": kind,
            "name": preview(name, self.secrets),
            "status": "running",
            **{k: preview(v, self.secrets) for k, v in fields.items()},
        }
        self.rows.append(row)
        self.starts[row["id"]] = time.perf_counter()
        self.notify()
        return row

    def finish(self, row, status, **fields):
        row.update(
            status=status,
            latency_ms=(time.perf_counter() - self.starts.pop(row["id"], time.perf_counter()))
            * 1000,
        )
        row.update({key: preview(value, self.secrets) for key, value in fields.items()})
        self.notify()

    def observe(self, event, fields):
        if not self.active:
            return
        if event == "agent_call_start":
            self.start("model", "Model", role=fields.get("role", "single"))
        elif event == "agent_call":
            row = next(
                (
                    row
                    for row in reversed(self.rows)
                    if row["kind"] == "model" and row["status"] == "running"
                ),
                None,
            )
            if row:
                self.finish(
                    row,
                    fields.get("status", "success"),
                    **{
                        key: fields.get(key)
                        for key in ("input_tokens", "output_tokens", "latency_ms")
                    },
                )
        elif event == "verification" and fields.get("tool_name"):
            row = self.start(
                "verification",
                fields["tool_name"],
                arguments=fields.get("arguments"),
                operation_id=fields.get("operation_id"),
            )
            category = fields.get("status")
            self.finish(
                row,
                "allowed"
                if category == "allowed"
                else "waiting"
                if category == "confirmation"
                else "blocked",
                result={"category": category},
            )

    def end(self, status="error"):
        for row in self.rows:
            if row["turn"] == self.turn and row["status"] == "running":
                self.finish(row, status)
        self.callback, self.active = None, False
