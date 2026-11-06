from casepilot.reliability.operations import PROVEN_UNCOMMITTED


def recovery_decision(strategy, *, read_only, state, proof, attempt, max_retries):
    if strategy not in {"S0", "S1", "S2"}:
        raise ValueError("Unknown recovery strategy")
    if strategy == "S0":
        return "stop"
    if strategy == "S2" and state == "committed" and proof == "receipt":
        return "receipt"
    if attempt > max_retries:
        return "stop"
    if read_only or (state == "failed" and proof in PROVEN_UNCOMMITTED):
        return "retry"
    return "stop"
