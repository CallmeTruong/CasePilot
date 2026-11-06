from casepilot.reliability.retry import recovery_decision


def reconcile(journal, episode_id, operation_id, *, strategy, read_only, attempt, max_retries):
    record = journal.get(episode_id, operation_id)
    state = record["state"] if record else "unknown"
    proof = record["proof"] if record else None
    decision = recovery_decision(
        strategy,
        read_only=read_only,
        state=state,
        proof=proof,
        attempt=attempt,
        max_retries=max_retries,
    )
    return decision, record
