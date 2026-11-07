"""Run with: uv run streamlit run dashboard/app.py"""

import streamlit as st

from dashboard.pages.evaluation import render as evaluation
from dashboard.pages.simulation import render as simulation

st.set_page_config(page_title="CasePilot Lite", layout="wide")
st.title("CasePilot Lite")
st.caption("Retail agents · reproducible development experiments")
simulate, evaluate = st.tabs(["Simulation", "Evaluation"])
with simulate:
    simulation()
with evaluate:
    evaluation()
