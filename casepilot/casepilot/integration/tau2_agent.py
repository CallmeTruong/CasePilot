import uuid

from tau2.agent.base_agent import HalfDuplexAgent, is_valid_agent_history_message

from casepilot.agents.single_agent import SingleAgentGraph
from casepilot.agents.state import CaseState
from casepilot.config import RunConfig


class CasePilotAgent(HalfDuplexAgent[CaseState]):
    def __init__(
        self,
        tools,
        domain_policy,
        config: RunConfig,
        *,
        episode_id=None,
        generate=None,
        trace=None,
        checkpoint_manager=None,
        memory_store=None,
        memory_scope=None,
        case_reference=None,
        customer_instruction=None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.config = config
        self.episode_id = episode_id or uuid.uuid4().hex
        self.case_reference = case_reference or self.episode_id
        self.current_state = None
        self.checkpoint_manager = checkpoint_manager
        self.backend_hash = None
        if config.architecture == "multi":
            from casepilot.agents.graph import MultiAgentGraph

            graph_class = MultiAgentGraph
        else:
            graph_class = SingleAgentGraph
        self.engine = graph_class(
            tools,
            domain_policy,
            config,
            generate=generate,
            trace=trace,
            checkpointer=checkpoint_manager.saver if checkpoint_manager else None,
            memory_store=memory_store,
            memory_scope=memory_scope,
            customer_instruction=customer_instruction,
        )

    def get_init_state(self, message_history=None):
        state = CaseState(episode_id=self.episode_id, case_id=self.case_reference)
        for message in message_history or []:
            if not is_valid_agent_history_message(message):
                raise ValueError("Non-observable message in initial history")
            if getattr(message, "role", None) == "assistant":
                for call in message.tool_calls or []:
                    state.pending_tool_calls[call.id] = {
                        "name": call.name,
                        "arguments": call.arguments,
                    }
                state.messages.append(message)
            else:
                self.engine.observe({"case": state, "incoming": message, "reply": None})
        self.current_state = state
        return state

    def generate_next_message(self, message, state):
        from tau2.data_model.message import UserMessage

        if isinstance(message, UserMessage) and (message.is_audio or message.is_tool_call()):
            raise ValueError("CasePilot Retail accepts observable text user messages only")
        if state.episode_id != self.episode_id:
            raise ValueError("Episode mismatch")
        reply, state = self.engine.invoke(message, state)
        self.current_state = state
        if self.checkpoint_manager and self.backend_hash:
            self.checkpoint_manager.save(state, self.backend_hash())
        return reply, state

    def set_seed(self, seed):
        self.config = RunConfig.model_validate(self.config.model_dump() | {"seed": seed})
        self.engine.config = self.config
        if self.checkpoint_manager:
            self.checkpoint_manager.config_hash = self.config.fingerprint()

    @classmethod
    def is_stop(cls, message):
        return "###STOP###" in (message.content or "")
