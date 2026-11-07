"""Customer support entrypoint. Research is in dashboard/research_app.py."""

import streamlit as st

from dashboard.customer import render

st.set_page_config(
    page_title="CasePilot Workspace",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render()
