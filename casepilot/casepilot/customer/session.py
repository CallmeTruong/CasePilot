"""One direct human conversation. No benchmark task, user simulator or evaluator."""

import threading
import time
import uuid
from copy import deepcopy
from weakref import finalize

from langsmith import tracing_context

from casepilot.agents.state import fingerprint as operation_fingerprint
from casepilot.customer.activity import ActivityRecorder
from casepilot.customer.language import fallback_language, localize
from casepilot.customer.observability import SessionTrace
from casepilot.customer.views import customer_text
from casepilot.security.guards import (
    SCOPE_MESSAGE,
    SECURITY_MESSAGE,
    CustomerSafetyGate,
    request_block,
    scrub_text,
)

UNKNOWN = (
    "I cannot confirm the outcome of this request and will not repeat it automatically. "
    "Please keep this conversation's support reference."
)


class CustomerSession:
    def __init__(self, agent, environment, settings):
        self.agent, self.environment, self.settings = agent, environment, settings
        self.case_reference = "CP-" + uuid.uuid4().hex[:8].upper()
        self.agent.case_reference = self.case_reference
        self.state = agent.get_init_state()
        self.transcript, self.receipts = [], {}
        self.traces, self.turns = [], []
        self.activity = []
        self.observer = ActivityRecorder(
            self.activity,
            [
                settings.api_key.get_secret_value(),
                settings.base_url.get_secret_value() if settings.base_url else "",
            ],
        )
        self.agent.engine.trace = SessionTrace(
            self.state.episode_id, self.traces, self.observer.observe
        )
        self.events = set()
        self.closed = self.busy = False
        self.lock = threading.RLock()
        self._cleanup = finalize(self, environment.close)
        self.confirmation_token = None
        self.language = "en"
        self.secrets = [
            settings.api_key.get_secret_value(),
            settings.base_url.get_secret_value() if settings.base_url else "",
        ]
        self.agent.engine.gate = CustomerSafetyGate(self.agent.engine.gate, self.receipts)

    def _event_id(self, event_id):
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 128:
            raise ValueError("Unable to send this request. Please try again.")
        if event_id not in self.events and len(self.events) >= 120:
            raise ValueError(
                "This session has reached its request limit. Please start a new session."
            )

    def _check(self):
        if self.closed:
            raise ValueError("This conversation has ended. You can start a new conversation.")
        if self.state.status in {"unsafe_unknown", "exhausted", "cancelled", "rejected"}:
            raise ValueError("This conversation has stopped safely. " + UNKNOWN)

    def send(self, text, event_id, *, on_activity=None):
        with self.lock:
            self._event_id(event_id)
            if event_id in self.events:
                return
            self._check()
            if not isinstance(text, str) or not text.strip() or len(text) > 6000:
                raise ValueError("Please enter a request between 1 and 6,000 characters.")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("Unable to send this request. Please try again.")
            self.events.add(event_id)
            self.confirmation_token = None
            if self.state.confirmation and self.state.confirmation.status in {
                "pending",
                "confirmed",
            }:
                self.state.confirmation.status = "invalidated"
            self.language = fallback_language(text, self.language)
            safe_text = scrub_text(text.strip(), self.secrets)
            self.transcript.append({"role": "user", "content": safe_text})
            self.agent.engine.gate.request = text.strip()
            blocked = "sensitive_data" if safe_text != text.strip() else request_block(text)
            if blocked:
                self._refuse(blocked, on_activity)
                return
            self.agent.engine.gate.observe_request(text.strip())
            self._run(text.strip(), on_activity=on_activity)

    def _refuse(self, reason, callback):
        if self.state.status == "waiting_confirmation":
            self.state.status = "active"
        self.observer.begin(callback)
        started = time.perf_counter()
        try:
            row = self.observer.start("security", "Scope & safety")
            self.observer.finish(row, "blocked", result={"category": reason})
            self.transcript[-1]["turn"] = self.observer.turn
            self.transcript.append(
                {
                    "role": "assistant",
                    "content": localize(
                        SCOPE_MESSAGE if reason == "scope" else SECURITY_MESSAGE, self.language
                    ),
                    "turn": self.observer.turn,
                }
            )
            self.turns.append(
                {
                    "turn": self.observer.turn,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "llm_calls": 0,
                    "tool_calls": 0,
                    "status": "blocked",
                }
            )
        finally:
            self.observer.end()

    def decide(self, approve, fingerprint, event_id, *, on_activity=None):
        with self.lock:
            self._event_id(event_id)
            if event_id in self.events:
                return
            self._check()
            confirmation = self.state.confirmation
            if (
                not confirmation
                or confirmation.status != "pending"
                or not self.confirmation_token
                or self.confirmation_token != fingerprint
                or confirmation.episode_id != self.state.episode_id
                or confirmation.plan_revision != self.state.plan_revision
                or confirmation.fingerprint
                != operation_fingerprint(
                    confirmation.tool_name,
                    confirmation.arguments,
                    self.state.customer_id,
                    self.state.plan_revision,
                    self.state.episode_id,
                )
            ):
                raise ValueError(
                    "The confirmation details have changed. Please review the latest request."
                )
            if not isinstance(approve, bool) or not isinstance(event_id, str) or not event_id:
                raise ValueError("Unable to confirm this request.")
            self.events.add(event_id)
            self.confirmation_token = None
            self.transcript.append(
                {
                    "role": "user",
                    "content": (
                        "I confirm the action above." if approve else "Do not perform this action."
                    ),
                }
            )
            self.agent.engine.emit(
                "verification", status="customer_confirmed" if approve else "customer_rejected"
            )
            if approve:
                self._run("yes", on_activity=on_activity)
            else:
                from casepilot.messages import UserMessage

                self.agent.engine.gate.observe_user("no", self.state)
                self.state.messages.append(UserMessage(role="user", content="no"))
                confirmation.status = "invalidated"
                self.state.status = "active"
                self.transcript.append(
                    {
                        "role": "assistant",
                        "content": localize(
                            "I have not made any changes. What would you like to adjust?",
                            self.language,
                        ),
                    }
                )

    def _run(self, text, *, on_activity=None):
        from casepilot.messages import UserMessage

        incoming = UserMessage(role="user", content=text)
        self.observer.begin(on_activity)
        transcript_start = len(self.transcript) - 1
        self.busy = True
        started = time.perf_counter()
        counters = (
            self.state.input_tokens,
            self.state.output_tokens,
            self.state.llm_calls,
            self.environment.dispatch_count,
        )
        trace_start = len(self.traces)
        before_commits = sum(self.environment.mutation_commits.values())
        before_errors = len(self.state.errors)
        outcome = None
        try:
            with tracing_context(enabled=False):
                for _ in range(self.agent.config.budgets.max_graph_steps):
                    reply, self.state = self.agent.generate_next_message(incoming, self.state)
                    if not reply.tool_calls:
                        if self.state.status == "unsafe_unknown":
                            content = UNKNOWN
                        elif self.state.status in {"exhausted", "rejected", "cancelled"}:
                            content = (
                                "I need to stop here to keep your request safe. Please keep"
                                " the support reference shown beside this conversation."
                            )
                        elif (
                            self.state.confirmation and self.state.confirmation.status == "pending"
                        ):
                            self.confirmation_token = uuid.uuid4().hex
                            content = (
                                "I have prepared the request. Please review the "
                                "confirmation below before proceeding."
                            )
                        elif len(self.state.errors) > before_errors:
                            content = self._gate_message(reply.content)
                        else:
                            content = customer_text(reply.content)
                        self.transcript.append(
                            {"role": "assistant", "content": scrub_text(content, self.secrets)}
                        )
                        return
                    call = reply.tool_calls[0]
                    activity = self.observer.start(
                        "tool", call.name, arguments=call.arguments, operation_id=call.id
                    )
                    tool_started = time.perf_counter()
                    incoming = self.environment.execute(call)
                    self.observer.finish(
                        activity, "error" if incoming.error else "success", result=incoming.content
                    )
                    self.agent.engine.emit(
                        "tool_dispatch",
                        tool_name=call.name,
                        operation_id=call.id,
                        latency_ms=(time.perf_counter() - tool_started) * 1000,
                        status="error" if incoming.error else "success",
                    )
                    self.receipts[call.id] = {
                        "tool_name": call.name,
                        "arguments": deepcopy(call.arguments),
                        "result": incoming.model_dump(mode="json"),
                    }
                self.state.status = "exhausted"
                self.transcript.append(
                    {
                        "role": "assistant",
                        "content": (
                            "This request needs too many verification steps. I stopped "
                            "safely to avoid unintended actions."
                        ),
                    }
                )
        except Exception as exc:
            outcome = "error"
            self.agent.engine.emit("error", error_category=type(exc).__name__)
            uncertain = bool(self.state.pending_tool_calls) or (
                sum(self.environment.mutation_commits.values()) > before_commits
            )
            if uncertain:
                self.state.status = "unsafe_unknown"
            if self.state.confirmation and self.state.confirmation.status == "confirmed":
                self.state.confirmation.status = "invalidated"
            self.transcript.append(
                {
                    "role": "assistant",
                    "content": UNKNOWN
                    if uncertain
                    else (
                        "The assistant is temporarily unavailable. Your messages are "
                        "still here; you can send a new message when the connection is "
                        "stable."
                    ),
                }
            )
        finally:
            self.observer.end("unknown" if self.state.status == "unsafe_unknown" else "error")
            for item in self.transcript[transcript_start:]:
                item["turn"] = self.observer.turn
                if item["role"] == "assistant":
                    item["content"] = localize(item["content"], self.language)
            attempts = [row for row in self.traces[trace_start:] if row["event"] == "agent_call"]
            turn_usage_known = all(
                row.get("input_tokens") is not None and row.get("output_tokens") is not None
                for row in attempts
            )
            self.turns.append(
                {
                    "turn": self.observer.turn,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "input_tokens": self.state.input_tokens - counters[0]
                    if turn_usage_known
                    else None,
                    "output_tokens": self.state.output_tokens - counters[1]
                    if turn_usage_known
                    else None,
                    "llm_calls": self.state.llm_calls - counters[2],
                    "tool_calls": self.environment.dispatch_count - counters[3],
                    "status": outcome or self.state.status,
                }
            )
            self.busy = False

    def _gate_message(self, original):
        category = self.state.errors[-1].get("category")
        if category == "authorization":
            if not self.state.customer_id:
                return (
                    "Please provide your order email, or your full name and ZIP code, so I "
                    "can verify your details."
                )
            return (
                "I can only help with information and orders belonging to the customer "
                "verified in this conversation."
            )
        if category == "state_conflict":
            return "The order information has changed. I need to check it again before continuing."
        if category == "scope":
            return SCOPE_MESSAGE
        if category == "identity_evidence":
            return (
                "Please provide your order email, or your full name and ZIP code, to verify"
                " your identity."
            )
        if category == "resource_limit":
            return (
                "This calculation exceeds safe limits. I can only perform simple "
                "calculations based on order data."
            )
        if category in {"schema", "duplicate", "parallel_tool_proposal", "handoff"}:
            return (
                "I need to check the request details before continuing to avoid an "
                "incorrect action."
            )
        return customer_text(original)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            try:
                self._cleanup()
            finally:
                self.state.customer_id = None
                self.state.owned_order_ids.clear()
                self.state.confirmation = None
                self.confirmation_token = None
                self.state.messages.clear()
                self.state.pending_tool_calls.clear()
                self.state.required_refresh_order_ids.clear()
                self.state.tool_results.clear()
                self.state.evidence.clear()
                self.state.plan = self.state.intent = None
                self.receipts.clear()
                self.transcript.clear()
                self.traces.clear()
                self.turns.clear()
                self.activity.clear()
                self.observer.end()
