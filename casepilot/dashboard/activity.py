"""Render observed execution, live and from saved conversation turns."""

import streamlit as st

STATUS = {
    "running": "Processing",
    "success": "Completed",
    "error": "Error",
    "unknown": "Outcome unknown",
    "allowed": "Allowed",
    "blocked": "Blocked",
    "waiting": "Awaiting confirmation",
}
KINDS = {"model": "Model", "tool": "Tool", "verification": "Verification", "security": "Safety"}


def render_steps(rows):
    for row in rows:
        status = STATUS.get(row["status"], row["status"])
        duration = (
            f" · {row['latency_ms'] / 1000:.2f}s" if row.get("latency_ms") is not None else ""
        )
        st.text(f"{KINDS.get(row['kind'], row['kind'])} · {row['name']} · {status}{duration}")
        if row["kind"] == "model":
            tokens = [
                str(row.get(key)) if row.get(key) is not None else "—"
                for key in ("input_tokens", "output_tokens")
            ]
            st.caption(f"{row.get('role', 'single')} · Tokens in {tokens[0]} / out {tokens[1]}")
        if "arguments" in row or "result" in row:
            with st.expander("Input / result · " + str(row["id"])):
                if row.get("arguments") is not None:
                    st.caption("Input")
                    st.json(row["arguments"])
                if row.get("result") is not None:
                    st.caption("Observed result · bounded preview")
                    st.json(row["result"])


def render_turn(session, turn):
    rows = [row for row in getattr(session, "activity", []) if row["turn"] == turn]
    if not rows:
        return
    latest = getattr(session, "turns", [])
    with st.expander(
        f"Agent activity · turn {turn} · {len(rows)} steps",
        expanded=bool(latest and latest[-1]["turn"] == turn),
    ):
        render_steps(rows)
    metric = next((r for r in getattr(session, "turns", []) if r["turn"] == turn), None)
    if metric:
        tokens = (
            "—"
            if metric["input_tokens"] is None or metric["output_tokens"] is None
            else f"{metric['input_tokens']} / {metric['output_tokens']}"
        )
        st.caption(
            f"{metric['latency_ms'] / 1000:.2f}s · {metric['llm_calls']} model · "
            f"{metric['tool_calls']} tool · Tokens in/out {tokens}"
        )
