from datetime import UTC, datetime

import pytest

from casepilot.memory.episodic import CaseHistory, HistoryStore


def history(customer="customer", scenario="scenario"):
    return CaseHistory(
        history_id="history",
        memory_scenario_id=scenario,
        customer_id=customer,
        case_id="case",
        source_episode_id="episode",
        session_id="session",
        intent="profile_lookup",
        result="profile_verified",
        last_action="get_user_details",
        verified=True,
        verified_at=datetime.now(UTC),
        source_operation_ids=["op"],
        facts={"zip": "10001"},
    )


def test_history_requires_verified_customer_and_scenario_scope(tmp_path):
    store = HistoryStore(tmp_path / "memory.sqlite")
    store.put(history())
    assert len(store.get_for_verified("scenario", "customer", verified_customer_id="customer")) == 1
    assert store.get_for_verified("other", "customer", verified_customer_id="customer") == []
    assert store.get_for_verified("scenario", "other", verified_customer_id="other") == []
    with pytest.raises(PermissionError):
        store.get_for_verified("scenario", "customer", verified_customer_id=None)
    with pytest.raises(PermissionError):
        store.get_for_verified("scenario", "customer", verified_customer_id="other")
    store.close()


def test_duplicate_case_source_is_not_reinserted_and_unverified_facts_rejected(tmp_path):
    store = HistoryStore(tmp_path / "memory.sqlite")
    store.put(history())
    store.put(history())
    assert len(store.get_for_verified("scenario", "customer", verified_customer_id="customer")) == 1
    with pytest.raises(ValueError):
        store.put(history().model_copy(update={"verified": False}))
    store.close()
