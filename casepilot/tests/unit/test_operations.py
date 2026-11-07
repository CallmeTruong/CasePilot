import pytest

from casepilot.reliability.operations import OperationJournal


def test_journal_persists_commit_and_rejects_argument_collision(tmp_path):
    path = tmp_path / "ops.sqlite"
    journal = OperationJournal(path)
    assert journal.start("episode", "op", "tool", {"a": 1})["state"] == "pending"
    journal.finish("episode", "op", "committed", result={"ok": True}, proof="receipt")
    journal.close()
    restored = OperationJournal(path)
    assert restored.get("episode", "op")["result"] == {"ok": True}
    assert restored.start("episode", "op", "tool", {"a": 1})["state"] == "committed"
    with pytest.raises(ValueError, match="different arguments"):
        restored.start("episode", "op", "tool", {"a": 2})
    assert restored.get("another", "op") is None
    restored.close()


def test_crash_pending_is_unknown_and_not_retried(tmp_path):
    path = tmp_path / "ops.sqlite"
    first = OperationJournal(path)
    first.start("episode", "op", "mutation", {})
    first.close()
    restored = OperationJournal(path)
    assert restored.start("episode", "op", "mutation", {})["state"] == "unknown"
    restored.close()


def test_failed_retry_requires_noncommit_proof(tmp_path):
    journal = OperationJournal(tmp_path / "ops.sqlite")
    journal.start("ep", "op", "mutation", {})
    journal.finish("ep", "op", "failed", proof="not_dispatched")
    assert journal.start("ep", "op", "mutation", {})["attempt"] == 2
    journal.close()
