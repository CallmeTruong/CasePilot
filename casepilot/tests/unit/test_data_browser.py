from types import SimpleNamespace

import pytest


def test_pinned_database_is_complete_and_independent_of_provider(monkeypatch):
    from casepilot.customer.browser import load_initial, table_rows

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db = load_initial()
    assert [
        len(table_rows(db, table)) for table in ("users", "orders", "products", "variants")
    ] == [500, 1000, 50, 591]
    db["users"].clear()
    assert len(load_initial()["users"]) == 500
    with pytest.raises(ValueError):
        table_rows(db, "../../.env")


def test_search_filter_sort_and_relationships_use_actual_records():
    from casepilot.customer.browser import (
        load_initial,
        query_rows,
        record_context,
        record_detail,
        table_rows,
    )

    db = load_initial()
    orders = table_rows(db, "orders")
    pending = query_rows(
        orders, filters={"status": "pending"}, sort_by="total_usd", descending=True
    )
    assert pending and all(row["status"] == "pending" for row in pending)
    assert pending[0]["total_usd"] >= pending[-1]["total_usd"]
    row = pending[0]
    assert query_rows(orders, query=row["id"] + " " + row["email"].upper()) == [row]
    detail = record_detail(db, "orders", row["id"])
    assert detail["record"]["order_id"] == row["id"]
    assert {"table": "users", "id": row["user_id"]} in detail["related"]
    assert record_context(db, "orders", row["id"]).find(row["email"]) >= 0
    assert "payment_history" not in record_context(db, "orders", row["id"])
    with pytest.raises(ValueError):
        record_context(db, "orders", "nonexistent")


def test_live_snapshot_reflects_mutations_without_changing_initial(retail_environment):
    from casepilot.customer.browser import load_initial, snapshot

    order = next(o for o in retail_environment.tools.db.orders.values() if o.status == "pending")
    original = order.status
    projection = SimpleNamespace(
        snapshot=lambda: retail_environment.tools.db.model_dump(mode="json")
    )
    session = SimpleNamespace(environment=projection, closed=False)
    view = snapshot(session)
    view["orders"][order.order_id]["status"] = "cancelled"
    assert order.status == original
    order.status = "cancelled"
    assert snapshot(session)["orders"][order.order_id]["status"] == "cancelled"
    assert load_initial()["orders"][order.order_id]["status"] == "pending"


def test_project_sources_are_allowlisted_and_never_attach_gold():
    from casepilot.customer.browser import load_initial, project_sources, record_context

    sources = project_sources()
    assert len(sources["rag_eval"]) == 30
    assert len(sources["fault_cases"]) == 18
    assert len(sources["memory_cases"]) == 12
    assert sources["policy"] and all(row["text"] for row in sources["policy"])
    with pytest.raises(ValueError):
        record_context(load_initial(), "rag_eval", "0")
