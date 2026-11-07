"""A single dispatch boundary shared by native and MCP execution."""

import hashlib
import json
import time

from tau2.data_model.message import ToolCall, ToolMessage
from tau2.environment.environment import Environment

from casepilot.agents.verifier import MUTATION_TOOLS
from casepilot.reliability.fault_injector import PreExecutionTimeout
from casepilot.reliability.operations import OperationJournal
from casepilot.reliability.verifier import reconcile


class GuardedEnvironment(Environment):
    def __init__(
        self,
        environment,
        agent,
        *,
        backend="native",
        timeout_seconds=15,
        recovery="S2",
        max_retries=2,
        backoff_seconds=0.5,
        journal_path=".cache/operations.sqlite3",
        fault=None,
    ):
        super().__init__(
            domain_name=environment.get_domain_name(),
            policy=environment.get_policy(),
            tools=environment.tools,
            user_tools=environment.user_tools,
        )
        self.agent, self.backend = agent, backend
        self.dispatch_results = {}
        self._dispatcher_closed = False
        self.dispatch_count = self.transport_errors = self.recovery_steps = 0
        self.mutation_commits = {}
        self.mutation_events = []
        self.recovery, self.max_retries = recovery, max_retries
        self.backoff_seconds, self.fault = backoff_seconds, fault
        self.journal = OperationJournal(journal_path)
        self.agent.backend_hash = self.get_db_hash
        from casepilot.integration.tau2.tools import create_executor, describe_tools
        from casepilot.tools.definitions import ToolDefinition

        candidates = getattr(agent, "tools", ())
        self.tool_definitions = (
            tuple(candidates)
            if candidates and all(isinstance(tool, ToolDefinition) for tool in candidates)
            else describe_tools(environment.get_tools())
        )
        self._executor = create_executor(environment, self.tool_definitions)
        self.bridge = None
        if backend == "mcp":
            from casepilot.mcp.bridge import EpisodeBridge
            from casepilot.mcp.client import MCPToolClient

            self.bridge = EpisodeBridge(agent.episode_id, self._bridge_request)
            self.client = MCPToolClient(self.bridge, timeout_seconds=timeout_seconds)
        elif backend != "native":
            raise ValueError("Unknown tool backend")

    def get_tools(self):
        return self.tool_definitions

    def _emit(self, event, **fields):
        emit = getattr(self.agent.engine, "emit", None)
        if emit:
            emit(event, **fields)

    def get_response(self, message):
        signature = json.dumps([message.name, message.arguments], sort_keys=True)
        previous = self.journal.get(self.agent.episode_id, message.id)
        args_hash = hashlib.sha256(
            json.dumps(message.arguments, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if previous and (
            previous["args_hash"] != args_hash or previous["tool_name"] != message.name
        ):
            raise ValueError("Operation ID reused with different arguments")
        if (
            message.id in self.dispatch_results
            and self.dispatch_results[message.id][0] != signature
        ):
            raise ValueError("Operation ID reused with different arguments")
        from casepilot.mcp.tool_registry import ACTION_TOOLS

        kind = "action" if message.name in ACTION_TOOLS else "read"
        for attempt in range(1, self.max_retries + 2):
            try:
                if self.backend == "native":
                    result = self._native_response(message)
                else:
                    result = ToolMessage.model_validate(
                        self.client.call_sync(kind, message.name, message.arguments, message.id)
                    )
                if result.error and self.fault and self.fault.triggered and self.recovery != "S2":
                    self.agent.current_state.status = "exhausted"
                    self.agent.current_state.termination_reason = "fault_no_recovery"
                return result
            except Exception:
                self.transport_errors += 1
                decision, record = reconcile(
                    self.journal,
                    self.agent.episode_id,
                    message.id,
                    strategy=self.recovery,
                    read_only=message.name not in MUTATION_TOOLS,
                    attempt=attempt,
                    max_retries=self.max_retries,
                )
                if decision == "receipt":
                    self.recovery_steps += 1
                    self._emit("verification", operation_id=message.id, status="committed_receipt")
                    return ToolMessage.model_validate(record["result"])
                if decision == "retry":
                    self.recovery_steps += 1
                    self._emit("retry", operation_id=message.id, attempt=attempt)
                    time.sleep(self.backoff_seconds * 2 ** (attempt - 1))
                    continue
                state = self.agent.current_state
                state.status = (
                    "exhausted" if record and record["state"] == "failed" else "unsafe_unknown"
                )
                state.termination_reason = "tool_recovery_stopped"
                if record and record["state"] == "pending":
                    self.journal.finish(state.episode_id, message.id, "unknown")
                return ToolMessage(
                    id=message.id,
                    role="tool",
                    content=(
                        "This operation failed or its outcome is unknown. "
                        "Stopped safely; do not retry blindly."
                    ),
                    error=True,
                )
        raise RuntimeError("Retry budget exhausted")

    def _bridge_request(self, payload):
        from casepilot.mcp.tool_registry import describe_tools

        available = describe_tools(self, payload.get("kind", "read"))
        if payload["action"] == "list_tools":
            return {"tools": available}
        names = {tool["name"] for tool in available}
        if payload["action"] != "call_tool" or payload.get("name") not in names:
            return {"protocol_error": "tool_group_mismatch"}
        message = ToolCall(
            id=payload["operation_id"], name=payload["name"], arguments=payload["arguments"]
        )
        return self._native_response(message).model_dump(mode="json")

    def _native_response(self, message):
        state = self.agent.current_state
        signature = json.dumps([message.name, message.arguments], sort_keys=True)
        if message.id in self.dispatch_results:
            previous_signature, result = self.dispatch_results[message.id]
            if signature != previous_signature:
                raise ValueError("Operation ID reused with different arguments")
            return result.model_copy(deep=True)
        if state is None or state.episode_id != self.agent.episode_id:
            raise ValueError("Dispatch has no matching episode state")
        verdict = self.agent.engine.gate.check(message.name, message.arguments, state)
        if not verdict.allowed:
            return ToolMessage(
                id=message.id,
                role="tool",
                content=verdict.message,
                error=True,
                execution_status="rejected",
            )
        record = self.journal.start(state.episode_id, message.id, message.name, message.arguments)
        if record["state"] == "committed":
            return ToolMessage.model_validate(record["result"])
        if record["state"] != "pending":
            raise TimeoutError("Previous operation execution is unresolved")
        if self.fault:
            try:
                self.fault.before(
                    self, message, attempt=record["attempt"], step=self.dispatch_count + 1
                )
            except PreExecutionTimeout:
                self.journal.finish(
                    state.episode_id,
                    message.id,
                    "failed",
                    proof="not_dispatched",
                    backend_hash=self.get_db_hash(),
                )
                raise
        before_hash = self.get_db_hash()
        self.dispatch_count += 1
        try:
            from langsmith import tracing_context

            with tracing_context(enabled=False):
                content = self._executor.invoke(message.name, message.arguments)
            self.sync_tools()
            result = ToolMessage(
                id=message.id, role="tool", content=content, requestor=message.requestor
            )
        except Exception as exc:
            result = ToolMessage(
                id=message.id,
                role="tool",
                content=f"Error: {exc}",
                requestor=message.requestor,
                error=True,
            )
        after_hash = self.get_db_hash()
        execution_state = "committed"
        proof = "receipt"
        if result.error:
            execution_state = "failed" if before_hash == after_hash else "unknown"
            proof = "backend_state_unchanged" if before_hash == after_hash else None
            if execution_state == "unknown":
                state.status = "unsafe_unknown"
                state.termination_reason = "partial_backend_mutation"
        self.journal.finish(
            state.episode_id,
            message.id,
            execution_state,
            result=result.model_dump(mode="json"),
            proof=proof,
            backend_hash=after_hash,
        )
        self.dispatch_results[message.id] = (signature, result.model_copy(deep=True))
        if not result.error and message.name in MUTATION_TOOLS:
            self.mutation_commits[message.id] = self.mutation_commits.get(message.id, 0) + 1
        if message.name in MUTATION_TOOLS:
            resource = message.arguments.get("order_id", message.arguments.get("user_id", ""))
            self.mutation_events.append(
                {
                    "operation_id": message.id,
                    "tool_name": message.name,
                    "resource_hash": hashlib.sha256(resource.encode()).hexdigest(),
                    "committed": not result.error,
                    "possibly_mutated": result.error and before_hash != after_hash,
                    "after_fault": bool(self.fault and self.fault.triggered),
                }
            )
        manager = getattr(self.agent, "checkpoint_manager", None)
        if manager:
            manager.save(state, after_hash)
        if self.fault and not result.error:
            self.fault.after(self, message, attempt=record["attempt"], step=self.dispatch_count)
        return result

    def close(self):
        if self._dispatcher_closed:
            return
        self._dispatcher_closed = True
        callbacks = []
        if self.bridge is not None:
            callbacks.append(self.bridge.close)
        callbacks.append(self.journal.close)
        manager = getattr(self.agent, "checkpoint_manager", None)
        if manager:
            callbacks.append(manager.close)
        close_agent = getattr(self.agent, "close", None)
        if close_agent:
            callbacks.append(close_agent)
        errors = []
        for close in callbacks:
            try:
                close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
