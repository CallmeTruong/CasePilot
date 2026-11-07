from casepilot.evaluation.recovery_metrics import summarize_recovery


def test_untriggered_and_missing_controls_are_not_successes():
    result = summarize_recovery([{"events": []}])
    assert result["recovery_success"].value is None
    assert result["fault_induced_drop"] is None
    result = summarize_recovery(
        [
            {"events": [1], "recovered": True},
            {"events": [1], "unsafe_recovery": True, "duplicate_commits": 2},
        ],
        paired_rewards=[(1, 0)],
        paired_latency=[(10, 15)],
    )
    assert result["recovery_success"].value == 0.5
    assert result["duplicate_commits"] == 2
    assert result["fault_induced_drop"] == 1
    assert result["latency_overhead_ms"] == 5
    assert result["paired_reward_count"] == result["paired_latency_count"] == 1
    assert result["duplicate_commit_rate"].value is None


def test_other_order_read_does_not_prove_fault_target_refresh():
    import hashlib
    import json

    from casepilot.evaluation.recovery_metrics import audit_target_refresh

    artifact = {
        "events": [
            {
                "fault_type": "F3",
                "operation_id": "mutation",
                "resource_hash": hashlib.sha256(b"#TARGET").hexdigest(),
            }
        ],
        "recovered": True,
    }
    operations = [
        {"operation_id": "mutation", "tool_name": "cancel_pending_order", "state": "failed"},
        {"operation_id": "read", "tool_name": "get_order_details", "state": "committed"},
    ]
    message = {
        "role": "tool",
        "id": "read",
        "error": False,
        "content": json.dumps({"order_id": "#OTHER"}),
    }
    assert not audit_target_refresh(artifact, operations, [message])["recovered"]
    message["content"] = json.dumps({"order_id": "#TARGET"})
    assert audit_target_refresh(artifact, operations, [message])["recovered"]
    assert artifact["recovered"]  # Original artifacts stay unchanged.
