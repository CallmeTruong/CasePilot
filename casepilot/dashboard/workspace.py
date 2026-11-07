"""Read-only project database, selection and linked records."""

import hashlib
import json
import math

import streamlit as st

from casepilot.customer.browser import (
    TABLES,
    load_initial,
    project_sources,
    query_rows,
    record_context,
    record_detail,
    snapshot,
    table_rows,
)
from casepilot.customer.views import status_label

SOURCES = {
    "retail": "Retail database",
    "policy": "RAG policy",
    "rag_eval": "RAG evaluation",
    "fault_cases": "Fault scenarios",
    "memory_cases": "Memory scenarios",
}
LABELS = {
    "id": "Record ID",
    "user_id": "Customer ID",
    "email": "Email",
    "status": "Status",
    "items": "Items",
    "total_usd": "Item total · USD",
    "name": "Name",
    "city": "City",
    "zip": "ZIP",
    "orders": "Orders",
    "variants": "SKUs",
    "product_id": "Product ID",
    "available": "In stock",
    "price_usd": "Catalog price · USD",
    "options": "Options",
}


def navigate(table, record_id):
    st.session_state["browser_table"] = table
    st.session_state["browser_query"] = record_id


def attach(db, table, record_id):
    st.session_state["attached_record"] = {
        "table": table,
        "id": record_id,
        "content": record_context(db, table, record_id),
    }
    st.session_state["mobile_view"] = "Agent"


def render_workspace(session):
    st.subheader("Project data")
    source_col, origin_col = st.columns(2)
    with source_col:
        source = st.selectbox(
            "Data source", list(SOURCES), format_func=SOURCES.get, key="browser_source"
        )
    if source != "retail":
        render_project(source)
        return
    current = bool(session and not session.closed)
    with origin_col:
        origin = st.radio(
            "Data snapshot",
            ["Current session", "Initial data"] if current else ["Initial data"],
            horizontal=True,
            key="browser_origin" if current else "browser_origin_initial",
        )
    db = snapshot(session) if current and origin == "Current session" else load_initial()
    st.caption(
        "Read-only · Synthetic Retail · "
        + origin
        + (" · " + session.case_reference if current else " · agent not initialized")
    )
    with st.container(key="browser_counts"):
        counts = st.columns(4)
    for column, (key, label) in zip(counts, TABLES.items(), strict=True):
        column.metric(label, f"{len(table_rows(db, key)):,}")
    table_col, search_col, filter_col = st.columns([1, 2, 1])
    with table_col:
        table = st.selectbox(
            "Table", list(TABLES), index=1, format_func=TABLES.get, key="browser_table"
        )
    rows = table_rows(db, table)
    with search_col:
        query = st.text_input(
            "Search table", placeholder="Order ID, email, name, SKU…", key="browser_query"
        )
    filters = {}
    if table in {"orders", "variants"}:
        field = "status" if table == "orders" else "available"
        values = sorted({row[field] for row in rows})
        with filter_col:
            value = st.selectbox(
                "Filter status",
                [None, *values],
                format_func=lambda v: (
                    "All"
                    if v is None
                    else status_label(v)
                    if field == "status"
                    else "In stock"
                    if v
                    else "Out of stock"
                ),
                key=f"filter_{table}",
            )
        if value is not None:
            filters[field] = value
    columns = list(rows[0])
    sort_col, page_col = st.columns([3, 1])
    with sort_col, st.expander("Sorting & visible columns"):
        sort = st.selectbox(
            "Sort by", columns, format_func=lambda k: LABELS.get(k, k), key=f"sort_{table}"
        )
        descending = st.checkbox("Descending", key=f"descending_{table}")
        shown = st.multiselect(
            "Visible columns",
            columns,
            default=columns,
            format_func=lambda k: LABELS.get(k, k),
            key=f"columns_{table}",
        )
    filtered = query_rows(rows, query, filters, sort, descending)
    st.caption(f"{len(filtered):,} / {len(rows):,} records · select a row to view details")
    if not filtered:
        st.info("No records match the filters.")
        return
    signature = hashlib.sha1(
        json.dumps(
            [table, origin, query, filters, sort, descending, [r["id"] for r in filtered]]
        ).encode()
    ).hexdigest()[:12]
    pages = max(1, math.ceil(len(filtered) / 20))
    with page_col:
        page = st.number_input(
            "Page", min_value=1, max_value=pages, value=1, step=1, key="page_" + signature
        )
    st.caption(f"Page {page} / {pages} · 20 rows per page")
    page_rows = filtered[(page - 1) * 20 : page * 20]
    live_view = st.empty()
    with st.container(key="browser_static"):
        selected = render_grid(db, table, page_rows, shown, signature, page)

    def refresh():
        active = st.session_state.get("customer_session")
        if not active or active.closed or (current and origin == "Initial data"):
            return
        updated = snapshot(active)
        updated_rows = query_rows(table_rows(updated, table), query, filters, sort, descending)
        with live_view.container():
            st.markdown(
                "<style>.st-key-browser_static{display:none}</style>", unsafe_allow_html=True
            )
            st.caption("Session data updated · agent is still working")
            st.dataframe(
                updated_rows[(page - 1) * 20 : page * 20],
                hide_index=True,
                width="stretch",
                height=355,
                column_config=LABELS,
            )
            if table == "orders":
                state = record_detail(updated, table, selected)["record"]["status"]
                st.caption(f"{selected} · {status_label(state)}")

    return refresh


def render_grid(db, table, page_rows, shown, signature, page):
    identifiers = [row["id"] for row in page_rows]
    grid_key = f"grid_{signature}_{page}"
    selection_key = f"record_{signature}_{page}"

    def select_row():
        selected = st.session_state[grid_key]["selection"]["rows"]
        if selected and 0 <= selected[0] < len(identifiers):
            st.session_state[selection_key] = identifiers[selected[0]]

    st.dataframe(
        page_rows,
        hide_index=True,
        width="stretch",
        height=355,
        column_order=["id", *[c for c in shown if c != "id"]],
        column_config=LABELS,
        on_select=select_row,
        selection_mode="single-row",
        key=grid_key,
    )
    selected = st.selectbox("Record", identifiers, key=selection_key)
    render_detail(db, table, selected)
    return selected


def render_detail(db, table, selected):
    detail = record_detail(db, table, selected)
    record = detail["record"]
    st.markdown(f"**{TABLES[table]} / {selected}**")
    if table == "orders":
        st.caption("Order status: " + status_label(record["status"]))
    st.button(
        "Attach record to chat",
        type="primary",
        on_click=attach,
        args=(db, table, selected),
        key="attach_record",
    )
    info, related, raw = st.tabs(["Details", "Related", "JSON"])
    with info:
        summary = next(row for row in table_rows(db, table) if row["id"] == selected)
        st.dataframe(
            [{"Field": LABELS.get(k, k), "Value": str(v)} for k, v in summary.items()],
            hide_index=True,
            width="stretch",
        )
        if table == "orders":
            st.markdown("**Order items · purchase prices**")
            st.dataframe(record["items"], hide_index=True, width="stretch")
            with st.expander("Address · payments · shipping"):
                st.json({k: record.get(k) for k in ("address", "payment_history", "fulfillments")})
        elif table == "users":
            st.json(record["address"])
    with related:
        links = list({(r["table"], r["id"]): r for r in detail["related"]}.values())
        if not links:
            st.caption("No related records.")
        for index, link in enumerate(links):
            st.button(
                f"{TABLES[link['table']]} → {link['id']}",
                key=f"related_{table}_{selected}_{index}",
                on_click=navigate,
                args=(link["table"], link["id"]),
            )
    with raw:
        st.json(record, expanded=False)


def render_project(source):
    rows = project_sources()[source]
    st.caption(
        "Project data · read-only · labels and answers are not automatically added to agent context"
    )
    query = st.text_input("Search source", key=f"project_query_{source}")
    rows = [
        row
        for row in rows
        if all(
            term in json.dumps(row, ensure_ascii=False).casefold()
            for term in query.casefold().split()
        )
    ]
    st.caption(f"{len(rows):,} records")
    if not rows:
        st.info("No matching records.")
        return
    summaries = [
        {
            "#": i + 1,
            "ID": row.get("id", row.get("scenario_id", row.get("chunk_id", str(i + 1)))),
            "Content": str(
                row.get(
                    "query",
                    row.get("section", row.get("description", row.get("fault_type", "Scenario"))),
                )
            ),
        }
        for i, row in enumerate(rows)
    ]
    st.dataframe(summaries, hide_index=True, width="stretch", height=355)
    index = st.selectbox(
        "Record details",
        range(len(rows)),
        format_func=lambda i: str(summaries[i]["ID"]),
        key=f"project_record_{source}_{hashlib.sha1(query.encode()).hexdigest()[:8]}",
    )
    if source == "policy":
        st.markdown(rows[index]["text"])
    with st.expander("JSON content", expanded=source != "policy"):
        st.json(rows[index])
