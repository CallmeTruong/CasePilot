"""Two-Agent graph with bounded structured planning and the same deterministic gate."""

from langgraph.graph import END, START, StateGraph

from casepilot.agents.executor import Executor
from casepilot.agents.planner import Planner, ResolutionPlan
from casepilot.agents.single_agent import SingleAgentGraph, Turn


class MultiAgentGraph(SingleAgentGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planner = Planner(self.call_model)
        self.executor = Executor(self.call_model)
        builder = StateGraph(Turn)
        builder.add_node("observe", self.observe)
        builder.add_node("plan", self.plan)
        builder.add_node("execute", self.decide)
        builder.add_node("verify", self.verify)
        builder.add_edge(START, "observe")
        builder.add_edge("observe", "plan")
        builder.add_edge("plan", "execute")
        builder.add_edge("execute", "verify")
        builder.add_edge("verify", END)
        self.graph = builder.compile(checkpointer=self.checkpointer)

    def plan(self, turn):
        state = turn["case"]
        if state.status in {"unsafe_unknown", "exhausted", "rejected", "cancelled"}:
            return {"case": state}
        if (
            state.graph_steps >= self.config.budgets.max_graph_steps
            or state.token_usage >= self.config.budgets.max_tokens
        ):
            state.status = "exhausted"
            state.termination_reason = "step_or_token_budget"
            return {"case": state}
        if state.confirmation and state.confirmation.status == "confirmed" and state.plan:
            # Execute the explicitly confirmed plan before asking the planner to revise it.
            return {"case": state}
        while True:
            if (
                state.graph_steps >= self.config.budgets.max_graph_steps
                or state.token_usage >= self.config.budgets.max_tokens
            ):
                state.status = "exhausted"
                state.termination_reason = "step_or_token_budget"
                break
            try:
                state.graph_steps += 1
                plan = self.planner.propose(state, self.prompt(state))
                allowed_ids = {e["chunk_id"] for e in state.evidence} | set(
                    state.memory_history_ids
                )
                if not set(plan.evidence_ids) <= allowed_ids:
                    raise ValueError("Planner cited unavailable evidence")
                proposed = plan.model_dump()
                material = ("intent", "actions", "requires_confirmation")
                if state.plan is not None and any(
                    state.plan.get(key) != proposed[key] for key in material
                ):
                    state.plan_revision += 1
                    if state.confirmation and state.confirmation.status in {"pending", "confirmed"}:
                        state.confirmation.status = "invalidated"
                state.plan = proposed
                state.intent = plan.intent
                needs_policy = plan.status == "blocked" or any(
                    "policy" in item.lower() or "evidence" in item.lower()
                    for item in plan.required_information
                )
                if (
                    self.rag
                    and needs_policy
                    and not plan.evidence_ids
                    and state.retrieval_rounds < 2
                ):
                    state.retrieval_rounds = 2
                    state.evidence = self.rag.search(
                        plan.intent + " " + " ".join(plan.actions), retrieval_round=2
                    )
                    self.emit(
                        "retrieval",
                        retrieval_strategy=self.config.policy_access,
                        evidence_ids=[e["chunk_id"] for e in state.evidence],
                    )
                    continue
                self.emit("handoff", role="planner", status=plan.status)
                break
            except ValueError:
                state.errors.append({"category": "invalid_plan"})
                if state.retry_count >= self.config.budgets.max_replans:
                    state.status = "rejected"
                    state.termination_reason = "invalid_plan_budget"
                    break
                state.retry_count += 1
        return {"case": state}

    def decide(self, turn):
        from tau2.data_model.message import AssistantMessage

        state = turn["case"]
        if state.required_refresh_order_ids and self.config.recovery_strategy == "S2":
            return super().decide(turn)
        if state.status in {"unsafe_unknown", "exhausted", "rejected", "cancelled"}:
            return {
                "case": state,
                "reply": AssistantMessage(
                    role="assistant",
                    content=(
                        "I cannot safely continue this case. "
                        "Please contact customer support. ###STOP###"
                    ),
                ),
            }
        plan = ResolutionPlan.model_validate(state.plan)
        state.graph_steps += 1
        reply = self.executor.propose(state, self.prompt(state), self.tools, plan)
        return {"case": state, "reply": reply}
