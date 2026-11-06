import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from casepilot.agents.state import CaseState


class CheckpointManager:
    def __init__(self, path, *, benchmark_revision, config_hash):
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        from langgraph.checkpoint.sqlite import SqliteSaver

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self.benchmark_revision, self.config_hash = benchmark_revision, config_hash
        self.connection.execute("""CREATE TABLE IF NOT EXISTS case_boundaries (
            episode_id TEXT NOT NULL, sequence INTEGER NOT NULL, benchmark_revision TEXT NOT NULL,
            config_hash TEXT NOT NULL, backend_hash TEXT NOT NULL, state_hash TEXT NOT NULL,
            state_json TEXT NOT NULL, PRIMARY KEY (episode_id, sequence))""")
        self.connection.commit()
        allowed = [("casepilot.agents.state", name) for name in ("CaseState", "Confirmation")]
        allowed += [
            ("tau2.data_model.message", name)
            for name in ("AssistantMessage", "UserMessage", "ToolMessage", "MultiToolMessage")
        ]
        serde = JsonPlusSerializer(
            pickle_fallback=False, allowed_msgpack_modules=allowed, allowed_json_modules=allowed
        )
        self.saver = SqliteSaver(self.connection, serde=serde)

    def save(self, state: CaseState, backend_hash: str):
        content = state.model_dump_json()
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self.lock, self.connection:
            sequence = self.connection.execute(
                "SELECT COALESCE(MAX(sequence), -1)+1 FROM case_boundaries WHERE episode_id=?",
                (state.episode_id,),
            ).fetchone()[0]
            self.connection.execute(
                "INSERT INTO case_boundaries VALUES (?,?,?,?,?,?,?)",
                (
                    state.episode_id,
                    sequence,
                    self.benchmark_revision,
                    self.config_hash,
                    backend_hash,
                    digest,
                    content,
                ),
            )
        return sequence

    def restore(
        self, episode_id, *, backend_hash, journal=None, benchmark_revision=None, config_hash=None
    ):
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM case_boundaries WHERE episode_id=? ORDER BY sequence DESC LIMIT 1",
                (episode_id,),
            ).fetchone()
        if row is None:
            raise ValueError("No checkpoint for this episode")
        expected = (benchmark_revision or self.benchmark_revision, config_hash or self.config_hash)
        if (row["benchmark_revision"], row["config_hash"]) != expected:
            raise ValueError("Checkpoint revision/config mismatch")
        if row["backend_hash"] != backend_hash:
            raise ValueError("Backend state was lost or changed; refusing checkpoint resume")
        if hashlib.sha256(row["state_json"].encode()).hexdigest() != row["state_hash"]:
            raise ValueError("Checkpoint integrity mismatch")
        state = CaseState.model_validate_json(row["state_json"])
        from pydantic import TypeAdapter
        from tau2.data_model.message import Message, ToolMessage

        state.messages = TypeAdapter(list[Message]).validate_python(
            json.loads(row["state_json"])["messages"]
        )
        if any(getattr(m, "role", None) == "system" for m in state.messages):
            raise ValueError("Checkpoint contains non-observable system history")
        receipts = []
        for operation_id in state.pending_tool_calls:
            record = journal.get(episode_id, operation_id) if journal else None
            if not record or record["state"] not in {"committed", "failed"} or not record["result"]:
                raise ValueError("Cannot resolve pending operation safely on resume")
            receipts.append(ToolMessage.model_validate(record["result"]))
        return state, receipts

    def close(self):
        self.connection.close()
