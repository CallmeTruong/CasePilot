"""Deterministic faults with evaluator-only timing/commit ground truth."""

import hashlib
from datetime import UTC, datetime


class PreExecutionTimeout(TimeoutError):
    pass


class PostCommitResponseLoss(TimeoutError):
    pass


class FaultInjector:
    def __init__(self, fault_type, target, *, seed=42, trigger_attempt=1):
        if fault_type not in {"F1", "F2", "F3"}:
            raise ValueError("Unknown fault type")
        self.fault_type, self.target = fault_type, target
        self.seed, self.trigger_attempt = seed, trigger_attempt
        self.triggered = False
        self.events = []

    def _record(self, environment, message, *, committed, step):
        self.triggered = True
        resource = message.arguments.get("order_id", message.arguments.get("user_id", ""))
        self.events.append(
            {
                "operation_id": message.id,
                "tool_name": message.name,
                "fault_type": self.fault_type,
                "seed": self.seed,
                "injection_step": step,
                "injected_at": datetime.now(UTC).isoformat(),
                "backend_committed": committed,
                "backend_state_hash": environment.get_db_hash(),
                "resource_hash": hashlib.sha256(resource.encode()).hexdigest(),
            }
        )

    def before(self, environment, message, *, attempt, step):
        if self.triggered or message.name != self.target or attempt != self.trigger_attempt:
            return
        if self.fault_type == "F1":
            self._record(environment, message, committed=False, step=step)
            raise PreExecutionTimeout("Dispatch was rejected before backend execution")
        if self.fault_type == "F3":
            order_id = message.arguments.get("order_id")
            if not order_id:
                raise ValueError("Stale-order faults require an order action")
            environment.tools.db.orders[order_id].status = "processed"
            self._record(environment, message, committed=False, step=step)

    def after(self, environment, message, *, attempt, step):
        if (
            not self.triggered
            and message.name == self.target
            and self.fault_type == "F2"
            and attempt == self.trigger_attempt
        ):
            self._record(environment, message, committed=True, step=step)
            raise PostCommitResponseLoss("Backend committed, response lost")
