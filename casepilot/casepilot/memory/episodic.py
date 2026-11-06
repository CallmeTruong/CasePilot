import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

from pydantic import AwareDatetime, Field

from casepilot.agents.verifier import IDENTITY_TOOLS, MUTATION_TOOLS
from casepilot.config import NonEmpty, StrictModel


class CaseHistory(StrictModel):
    history_id: NonEmpty
    memory_scenario_id: NonEmpty
    customer_id: NonEmpty
    case_id: NonEmpty
    source_episode_id: NonEmpty
    session_id: NonEmpty
    intent: NonEmpty
    result: NonEmpty
    last_action: NonEmpty
    verified: bool
    verified_at: AwareDatetime
    source_operation_ids: list[NonEmpty] = Field(min_length=1)
    facts: dict = Field(default_factory=dict)


class HistoryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute("""CREATE TABLE IF NOT EXISTS case_history (
            history_id TEXT PRIMARY KEY, scope TEXT NOT NULL, customer_id TEXT NOT NULL,
            case_id TEXT NOT NULL, source_episode_id TEXT NOT NULL, verified_at TEXT NOT NULL,
            payload TEXT NOT NULL, UNIQUE(scope, case_id, source_episode_id))""")
        self.connection.commit()

    def put(self, history: CaseHistory):
        if not history.verified or not history.source_operation_ids:
            raise ValueError("History must contain verified source operations")
        if set(history.facts) - {"zip", "last_order_id", "case_reference"}:
            raise ValueError("Unknown historical fact fields")
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO case_history VALUES (?,?,?,?,?,?,?)",
                (
                    history.history_id,
                    history.memory_scenario_id,
                    history.customer_id,
                    history.case_id,
                    history.source_episode_id,
                    history.verified_at.isoformat(),
                    history.model_dump_json(),
                ),
            )

    def get_for_verified(self, scope, customer_id, *, verified_customer_id, limit=5):
        if not verified_customer_id or customer_id != verified_customer_id:
            raise PermissionError("Verify the same customer before querying case history")
        if not 1 <= limit <= 20:
            raise ValueError("History retrieval limit must be 1..20")
        with self.lock:
            rows = self.connection.execute(
                "SELECT payload FROM case_history WHERE scope=? "
                "AND customer_id=? ORDER BY verified_at DESC, history_id LIMIT ?",
                (scope, customer_id, limit),
            ).fetchall()
        return [CaseHistory.model_validate_json(row[0]) for row in rows]

    def record_verified_case(self, *, scope, state, journal):
        if not state.customer_id:
            raise PermissionError("Cannot store an unverified customer's case")
        records = [
            record for record in journal.records(state.episode_id) if record["state"] == "committed"
        ]
        identity, profile, last = None, None, None
        for record in records:
            receipt = journal.get(state.episode_id, record["operation_id"])
            raw = (receipt["result"] or {}).get("content", "")
            try:
                content = json.loads(raw)
            except (ValueError, TypeError):
                content = raw
            if record["tool_name"] in IDENTITY_TOOLS and content == state.customer_id:
                identity = receipt
            if isinstance(content, dict) and content.get("user_id") == state.customer_id:
                if record["tool_name"] == "get_user_details":
                    profile = (receipt, content)
                if (
                    record["tool_name"] in MUTATION_TOOLS
                    or record["tool_name"] == "get_user_details"
                ):
                    last = receipt
        if identity is None or profile is None or last is None:
            raise ValueError("Case history requires observed identity and profile receipts")
        case_id = state.case_id or state.episode_id
        history = CaseHistory(
            history_id=uuid.uuid4().hex,
            memory_scenario_id=scope,
            customer_id=state.customer_id,
            case_id=case_id,
            source_episode_id=state.episode_id,
            session_id=state.episode_id,
            intent=state.intent or "profile_lookup",
            result="action_committed"
            if last["tool_name"] in MUTATION_TOOLS
            else "profile_verified",
            last_action=last["tool_name"],
            verified=True,
            verified_at=datetime.fromisoformat(last["updated_at"]),
            source_operation_ids=list(
                dict.fromkeys(
                    [identity["operation_id"], profile[0]["operation_id"], last["operation_id"]]
                )
            ),
            facts={"zip": profile[1].get("address", {}).get("zip"), "case_reference": case_id},
        )
        self.put(history)
        return history

    def close(self):
        self.connection.close()
