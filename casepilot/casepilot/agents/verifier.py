"""Deterministic invariants; policy eligibility remains in the model and upstream toolkit."""

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from jsonschema import Draft202012Validator, ValidationError

from casepilot.agents.state import CaseState, Confirmation, fingerprint
from casepilot.security.guards import validate_calculation

IDENTITY_TOOLS = frozenset({"find_user_id_by_email", "find_user_id_by_name_zip"})
MUTATION_TOOLS = frozenset(
    {
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
    }
)
SAFE_BEFORE_AUTH = IDENTITY_TOOLS | {"calculate", "transfer_to_human_agents"}


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    category: str = "allowed"
    message: str = ""


class PolicyGate:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    def schema_valid(self, name, args):
        tool = self.tools.get(name)
        if tool is None:
            return False
        schema = dict(tool.openai_schema["function"]["parameters"])
        schema["additionalProperties"] = False
        return Draft202012Validator(schema).is_valid(args)

    def check(self, name: str, args: dict, state: CaseState) -> GateResult:
        tool = self.tools.get(name)
        if tool is None:
            return GateResult(False, "schema", "This action is not available.")
        try:
            schema = dict(tool.openai_schema["function"]["parameters"])
            schema["additionalProperties"] = False
            Draft202012Validator(schema).validate(args)
        except ValidationError:
            return GateResult(False, "schema", "The requested action has invalid details.")
        if name == "calculate" and not validate_calculation(args.get("expression")):
            return GateResult(
                False, "resource_limit", "This calculation exceeds safe arithmetic limits."
            )
        if state.customer_id is None and name not in SAFE_BEFORE_AUTH:
            return GateResult(
                False,
                "authorization",
                "Please provide your email or name and ZIP "
                "code so I can verify your identity first.",
            )
        if state.customer_id and name in IDENTITY_TOOLS:
            return GateResult(
                False,
                "authorization",
                "I can only assist the customer already verified in this conversation.",
            )
        if "user_id" in args and args["user_id"] != state.customer_id:
            return GateResult(False, "authorization", "I can only access your own profile.")
        if "order_id" in args and args["order_id"] not in state.owned_order_ids:
            return GateResult(
                False,
                "authorization",
                "I need to look up your profile's orders before accessing that order.",
            )
        if name not in MUTATION_TOOLS:
            return GateResult(True)
        if args.get("order_id") in state.required_refresh_order_ids:
            return GateResult(
                False,
                "state_conflict",
                "The order state changed. I need to refresh its details before continuing.",
            )
        key = fingerprint(name, args, state.customer_id, state.plan_revision, state.episode_id)
        if state.completed_actions.get(key) == state.user_turn:
            return GateResult(False, "duplicate", "This action has already been completed.")
        existing = state.confirmation
        if (
            existing
            and existing.episode_id == state.episode_id
            and existing.fingerprint == key
            and existing.status == "confirmed"
        ):
            return GateResult(True)
        if not existing or existing.fingerprint != key or existing.status != "pending":
            if existing:
                existing.status = "invalidated"
            state.confirmation = Confirmation(
                episode_id=state.episode_id,
                fingerprint=key,
                tool_name=name,
                arguments=args.copy(),
                plan_revision=state.plan_revision,
                user_turn=state.user_turn,
            )
        state.status = "waiting_confirmation"
        details = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in args.items())
        action = name.replace("_", " ")
        return GateResult(
            False,
            "confirmation",
            f"I will {action} ({details}). "
            "Please confirm yes to proceed, or tell me what to change.",
        )

    def observe_user(self, content: str | None, state: CaseState) -> None:
        state.user_turn += 1
        confirmation = state.confirmation
        if confirmation and confirmation.status == "confirmed":
            confirmation.status = "invalidated"
        if not confirmation or confirmation.status != "pending":
            return
        # Conservative explicit confirmations, including common benchmark simulator phrasings.
        text = (content or "").strip().lower()
        affirmative = re.match(r"^(?:yes|please proceed|go ahead|confirmed|i confirm)\b", text)
        if re.search(
            r"\b(?:no|not|don't|but|however|instead|change|different|except|unless|wait)\b", text
        ):
            affirmative = None
        confirmation.status = "confirmed" if affirmative else "invalidated"
        if affirmative:
            confirmation.user_turn = state.user_turn
            confirmation.user_message_id = f"{state.episode_id}:user:{state.user_turn}"
            confirmation.confirmed_at = datetime.now(UTC)

    def observe_result(
        self, name: str, args: dict, content: str | None, error: bool, state: CaseState
    ) -> None:
        if error:
            state.errors.append({"category": "backend", "tool_name": name})
            if name in MUTATION_TOOLS and args.get("order_id") and "Non-" in (content or ""):
                state.required_refresh_order_ids.add(args["order_id"])
            if name in MUTATION_TOOLS and state.confirmation:
                state.confirmation.status = "invalidated"
            return
        try:
            value = json.loads(content or "null")
        except (ValueError, TypeError):
            value = content
        if name in IDENTITY_TOOLS and state.customer_id is None:
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]+", value):
                state.customer_id = value
        if name == "get_user_details" and isinstance(value, dict):
            if value.get("user_id") == state.customer_id:
                state.owned_order_ids = set(value.get("orders", []))
        if name == "get_order_details" and isinstance(value, dict):
            state.required_refresh_order_ids.discard(args.get("order_id"))
        if name in MUTATION_TOOLS:
            key = fingerprint(name, args, state.customer_id, state.plan_revision, state.episode_id)
            state.completed_actions[key] = state.user_turn
            if state.confirmation:
                state.confirmation.status = "consumed"
            state.status = "active"
