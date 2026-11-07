"""Local data browser, direct agent conversation and live execution."""

import uuid
from pathlib import Path

import streamlit as st

from casepilot.customer import factory
from casepilot.customer.views import confirmation_view
from dashboard.activity import render_steps, render_turn
from dashboard.workspace import render_workspace


def dispatch(text=None, *, decision=None, refresh=None):
    try:
        session = st.session_state.get("customer_session")
        if session is None:
            session = factory.create_session()
            st.session_state["customer_session"] = session
        if text:
            with st.chat_message("user"):
                st.text(text)
        with st.status("Agent is working…", expanded=True) as status:
            live = st.empty()

            def update(rows):
                turn = rows[-1]["turn"] if rows else None
                with live.container():
                    render_steps([row for row in rows if row["turn"] == turn])
                if rows and rows[-1]["kind"] == "tool" and rows[-1]["status"] != "running":
                    if refresh:
                        refresh()

            event_id = uuid.uuid4().hex
            if decision is not None:
                session.decide(decision[0], decision[1], event_id, on_activity=update)
            else:
                session.send(text, event_id, on_activity=update)
            failed = session.state.status in {"unsafe_unknown", "exhausted", "rejected"}
            status.update(
                label="Awaiting your confirmation"
                if session.state.status == "waiting_confirmation"
                else "Turn completed",
                state="error" if failed else "complete",
            )
    except Exception:
        st.session_state["customer_notice"] = (
            "Unable to process the request. Check the current session and confirmation "
            "before continuing."
        )


def new_session():
    session = st.session_state.pop("customer_session", None)
    if session:
        session.close()
    for key in ("attached_record", "customer_notice", "browser_origin"):
        st.session_state.pop(key, None)


def render_confirmation(session, refresh=None):
    view = confirmation_view(session.state, session.receipts)
    token = getattr(session, "confirmation_token", None)
    if not view or not token:
        return
    with st.container(border=True):
        st.subheader(view["title"])
        for label, value in view["details"]:
            st.caption(label)
            st.text(value)
        st.caption(view["notice"])
        if st.button("Confirm action", type="primary", key="approve_" + token, width="stretch"):
            dispatch(decision=(True, token), refresh=refresh)
            st.rerun()
        if st.button("Decline action", key="reject_" + token, width="stretch"):
            dispatch(decision=(False, token), refresh=refresh)
            st.rerun()


def render_chat(session, refresh=None):
    st.subheader("Agent")
    st.caption("Conversation · tools · execution results")
    if st.session_state.get("customer_notice"):
        st.warning(st.session_state.pop("customer_notice"))
    if not session or not session.transcript:
        st.info(
            "Select a record on the left to attach to chat, or enter a request. Model "
            "and tool activity will appear here."
        )
    if session:
        rendered = set()
        for item in session.transcript:
            turn = item.get("turn")
            if item["role"] == "assistant" and turn and turn not in rendered:
                render_turn(session, turn)
                rendered.add(turn)
            with st.chat_message(item["role"]):
                st.text(item["content"])
        render_confirmation(session, refresh)
        st.caption("Session: " + session.case_reference)
    attached = st.session_state.get("attached_record")
    if attached:
        st.caption(f"Attached: {attached['table']} / {attached['id']}")
        st.button(
            "Remove attachment", on_click=lambda: st.session_state.pop("attached_record", None)
        )
    disabled = bool(
        session
        and (
            session.busy
            or session.closed
            or session.state.status in {"unsafe_unknown", "exhausted", "rejected", "cancelled"}
        )
    )
    text = st.chat_input("Ask the agent…", max_chars=5000, disabled=disabled)
    if text:
        if attached:
            text += "\n\n" + attached["content"]
        dispatch(text, refresh=refresh)
        st.session_state.pop("attached_record", None)
        st.rerun()


def render():
    st.markdown(
        "<style>" + Path(__file__).with_name("styles.css").read_text(encoding="utf-8") + "</style>",
        unsafe_allow_html=True,
    )
    header, action = st.columns([5, 1])
    with header:
        st.title("CasePilot Workspace")
        st.caption("Project data & agent · Synthetic Retail · local workspace")
    with action:
        st.button("New session", on_click=new_session, width="stretch")
    with st.container(key="mobile_switch"):
        view = st.radio(
            "View",
            ["Data", "Agent"],
            horizontal=True,
            key="mobile_view",
            label_visibility="collapsed",
        )
    hide = "database_panel" if view == "Agent" else "chat_panel"
    st.markdown(
        f"<style>@media(max-width:768px){{[data-testid=stColumn]:has(.st-key-{hide}){{display:none!important}}}}</style>",
        unsafe_allow_html=True,
    )
    session = st.session_state.get("customer_session")
    database, agent = st.columns([1.85, 1], gap="large")
    with database:
        with st.container(border=True, key="database_panel"):
            refresh = render_workspace(session)
    with agent:
        with st.container(border=True, key="chat_panel"):
            render_chat(session, refresh)
