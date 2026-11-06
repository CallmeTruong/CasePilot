import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

PROVEN_UNCOMMITTED = frozenset({"not_dispatched", "backend_state_unchanged"})


class OperationJournal:
    def __init__(self, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS operations (
            episode_id TEXT NOT NULL, operation_id TEXT NOT NULL, tool_name TEXT NOT NULL,
            args_hash TEXT NOT NULL, state TEXT NOT NULL, attempt INTEGER NOT NULL,
            proof TEXT, result_json TEXT, backend_hash TEXT, created_at TEXT, updated_at TEXT,
            PRIMARY KEY (episode_id, operation_id))""")
        self.connection.commit()

    def get(self, episode_id, operation_id):
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM operations WHERE episode_id=? AND operation_id=?",
                (episode_id, operation_id),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["result"] = json.loads(result.pop("result_json")) if row["result_json"] else None
            return result

    def start(self, episode_id, operation_id, tool_name, arguments):
        digest = hashlib.sha256(
            json.dumps(arguments, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        now = datetime.now(UTC).isoformat()
        with self.lock, self.connection:
            previous = self.get(episode_id, operation_id)
            if previous is None:
                self.connection.execute(
                    "INSERT INTO operations VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        episode_id,
                        operation_id,
                        tool_name,
                        digest,
                        "pending",
                        1,
                        None,
                        None,
                        None,
                        now,
                        now,
                    ),
                )
            elif previous["args_hash"] != digest or previous["tool_name"] != tool_name:
                raise ValueError("Operation ID reused with different arguments")
            elif previous["state"] == "pending":
                self.connection.execute(
                    "UPDATE operations SET state='unknown', updated_at=? "
                    "WHERE episode_id=? AND operation_id=?",
                    (now, episode_id, operation_id),
                )
            elif previous["state"] == "failed" and previous["proof"] in PROVEN_UNCOMMITTED:
                self.connection.execute(
                    "UPDATE operations SET state='pending', attempt=attempt+1, "
                    "updated_at=? WHERE episode_id=? AND operation_id=?",
                    (now, episode_id, operation_id),
                )
        return self.get(episode_id, operation_id)

    def finish(
        self, episode_id, operation_id, state, *, result=None, proof=None, backend_hash=None
    ):
        if state not in {"committed", "failed", "unknown"}:
            raise ValueError("Invalid operation completion state")
        if state in {"committed", "failed"} and not proof:
            raise ValueError("Definite execution states require backend evidence")
        with self.lock, self.connection:
            previous = self.get(episode_id, operation_id)
            if previous is None:
                raise ValueError("Operation was not journaled before dispatch")
            if previous["state"] == "committed" and (
                state != "committed" or result != previous["result"]
            ):
                raise ValueError("Cannot overwrite a committed operation receipt")
            self.connection.execute(
                "UPDATE operations SET state=?, proof=?, result_json=?, "
                "backend_hash=?, updated_at=? WHERE episode_id=? AND operation_id=?",
                (
                    state,
                    proof,
                    json.dumps(result) if result is not None else None,
                    backend_hash,
                    datetime.now(UTC).isoformat(),
                    episode_id,
                    operation_id,
                ),
            )

    def records(self, episode_id):
        with self.lock:
            return [
                dict(row)
                for row in self.connection.execute(
                    "SELECT episode_id, operation_id, tool_name, state, attempt, proof, "
                    "backend_hash "
                    "FROM operations WHERE episode_id=? ORDER BY created_at",
                    (episode_id,),
                )
            ]

    def close(self):
        self.connection.close()
