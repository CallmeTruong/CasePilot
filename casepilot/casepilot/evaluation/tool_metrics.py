from casepilot.evaluation.metrics import ratio


def summarize_tools(
    *, proposals, schema_valid, dispatched, transport_errors, committed, duplicates
):
    return {
        "proposals": proposals,
        "dispatches": dispatched,
        "committed_mutations": committed,
        "schema_validity": ratio(schema_valid, proposals),
        "transport_error_rate": ratio(transport_errors, dispatched),
        "duplicate_commits": duplicates,
        "duplicate_commit_rate": ratio(duplicates, committed),
    }
