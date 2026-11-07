from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest


def app():
    return AppTest.from_file(Path("dashboard/app.py").resolve()).run(timeout=45)


def test_browser_landing_and_filters_do_not_initialize_provider(monkeypatch):
    import casepilot.customer.factory as factory

    def forbidden():
        raise AssertionError("Browsing must not initialize a provider")

    monkeypatch.setattr(factory, "create_session", forbidden)
    page = app()
    assert not page.exception
    assert any("Workspace" in title.value for title in page.title)
    assert len(page.chat_input) == 1
    assert any(metric.value == "1,000" for metric in page.metric)
    page.text_input(key="browser_query").set_value("not-present@example.test").run(timeout=45)
    assert not page.exception and "customer_session" not in page.session_state
    assert any("0 / 1,000" in caption.value for caption in page.caption)
    page.selectbox(key="browser_table").select("users").run(timeout=45)
    assert not page.exception and "customer_session" not in page.session_state


def test_attach_record_requires_explicit_send_and_rerender_never_replays(
    monkeypatch, retail_environment
):
    import casepilot.customer.factory as factory
    from casepilot.agents.state import CaseState

    class Stub:
        state = CaseState(episode_id="stub")
        receipts = {}
        transcript = []
        activity = []
        turns = []
        busy = closed = False
        case_reference = "CP-TEST"
        environment = SimpleNamespace(
            snapshot=lambda: retail_environment.tools.db.model_dump(mode="json")
        )

        def __init__(self):
            self.sent = []

        def send(self, text, event_id, *, on_activity=None):
            self.sent.append(text)
            self.transcript = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "Xin chào"},
            ]

        def close(self):
            self.closed = True

    stub = Stub()
    monkeypatch.setattr(factory, "create_session", lambda: stub)
    page = app()
    page.button(key="attach_record").click().run(timeout=45)
    assert not stub.sent and "customer_session" not in page.session_state
    assert page.session_state["attached_record"]["table"] == "orders"
    page.chat_input[0].set_value("Kiểm tra đơn này").run(timeout=45)
    assert not page.exception and len(stub.sent) == 1
    assert "order" in stub.sent[0] and "Kiểm tra đơn này" in stub.sent[0]
    assert page.radio(key="browser_origin").value == "Current session"
    page.run(timeout=45)
    page.selectbox(key="browser_table").select("products").run(timeout=45)
    assert not page.exception and len(stub.sent) == 1 and not stub.closed
    next(b for b in page.button if b.label == "New session").click().run(timeout=45)
    assert stub.closed and "customer_session" not in page.session_state


def test_project_dataset_inspection_cannot_attach_gold(monkeypatch):
    import casepilot.customer.factory as factory

    monkeypatch.setattr(factory, "create_session", lambda: (_ for _ in ()).throw(AssertionError()))
    page = app()
    page.selectbox(key="browser_source").select("rag_eval").run(timeout=45)
    assert not page.exception
    assert not any(b.key == "attach_record" for b in page.button)
    assert "customer_session" not in page.session_state


def test_record_dropdown_can_override_a_previous_grid_selection():
    page = app()
    grid = next(e for e in page.dataframe if e.key and e.key.startswith("grid_"))
    page.session_state[grid.key] = {"selection": {"rows": [1], "columns": [], "cells": []}}
    page.run(timeout=45)
    record = next(e for e in page.selectbox if e.key and e.key.startswith("record_"))
    requested = record.options[2]
    record.select(requested).run(timeout=45)
    record = next(e for e in page.selectbox if e.key and e.key.startswith("record_"))
    assert not page.exception and record.value == requested


def test_completed_tool_refreshes_live_table_before_send_returns(monkeypatch, retail_environment):
    import streamlit as st

    import casepilot.customer.factory as factory
    from casepilot.agents.state import CaseState

    order = next(o for o in retail_environment.tools.db.orders.values() if o.status == "pending")
    displayed = []
    original = st.dataframe

    def watch(data=None, *args, **kwargs):
        if isinstance(data, list):
            displayed.extend(row for row in data if isinstance(row, dict))
        return original(data, *args, **kwargs)

    monkeypatch.setattr(st, "dataframe", watch)

    class Stub:
        environment = SimpleNamespace(
            snapshot=lambda: retail_environment.tools.db.model_dump(mode="json")
        )
        state = CaseState(episode_id="live")
        transcript, receipts, activity, turns = [], {}, [], []
        closed = busy = False
        case_reference = "CP-LIVE"
        refreshed_before_return = False

        def send(self, text, event_id, *, on_activity=None):
            order.status = "cancelled"
            on_activity(
                [
                    {
                        "id": 1,
                        "turn": 1,
                        "kind": "tool",
                        "name": "cancel_pending_order",
                        "status": "success",
                        "latency_ms": 1,
                    }
                ]
            )
            self.refreshed_before_return = any(
                row.get("id") == order.order_id and row.get("status") == "cancelled"
                for row in displayed
            )

    stub = Stub()
    monkeypatch.setattr(factory, "create_session", lambda: stub)
    page = app()
    page.text_input(key="browser_query").set_value(order.order_id).run(timeout=45)
    page.chat_input[0].set_value("Xử lý yêu cầu").run(timeout=45)
    assert not page.exception and stub.refreshed_before_return
