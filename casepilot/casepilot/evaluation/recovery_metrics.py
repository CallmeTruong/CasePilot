"""Only triggered faults form recovery denominators; missing controls stay unknown."""

import hashlib
import json

from casepilot.evaluation.metrics import ratio


def audit_target_refresh(artifact, operations, conversation):
    """Require an actual successful read of the faulted order, not any later order read."""
    events = artifact.get("events", [])
    result = dict(artifact)
    if not events or events[0]["fault_type"] != "F3":
        return result
    event = events[0]
    target_index = next(
        (i for i, op in enumerate(operations) if op["operation_id"] == event["operation_id"]), None
    )
    receipts = {
        message.get("id"): message
        for message in conversation
        if message.get("role") == "tool" and not message.get("error")
    }
    refreshed = False
    for operation in operations[target_index + 1 :] if target_index is not None else []:
        if operation["tool_name"] != "get_order_details" or operation["state"] != "committed":
            continue
        receipt = receipts.get(operation["operation_id"], {})
        try:
            order_id = json.loads(receipt.get("content") or "null").get("order_id")
        except (ValueError, AttributeError, TypeError):
            continue
        if (
            isinstance(order_id, str)
            and hashlib.sha256(order_id.encode()).hexdigest() == event["resource_hash"]
        ):
            refreshed = True
    result["fresh_read_target_verified"] = refreshed
    result["recovered"] = bool(result.get("recovered") and refreshed)
    return result


def summarize_recovery(artifacts, *, paired_rewards=None, paired_latency=None):
    triggered = [a for a in artifacts if a.get("events")]
    duplicate_count = sum(a.get("duplicate_commits", 0) for a in triggered)
    committed_count = sum(
        bool(m.get("committed")) for a in triggered for m in a.get("mutations", [])
    )
    return {
        "scheduled": len(artifacts),
        "triggered": len(triggered),
        "recovery_success": ratio(sum(bool(a.get("recovered")) for a in triggered), len(triggered)),
        "unsafe_recovery": ratio(
            sum(bool(a.get("unsafe_recovery")) for a in triggered), len(triggered)
        ),
        "duplicate_commits": duplicate_count,
        "committed_mutations": committed_count,
        "duplicate_commit_rate": ratio(duplicate_count, committed_count),
        "paired_reward_count": len(paired_rewards or []),
        "paired_latency_count": len(paired_latency or []),
        "fault_induced_drop": None
        if not paired_rewards
        else sum(a - b for a, b in paired_rewards) / len(paired_rewards),
        "latency_overhead_ms": None
        if not paired_latency
        else sum(b - a for a, b in paired_latency) / len(paired_latency),
        "paired_undefined_reason": None if paired_rewards else "no matched no-fault controls",
        "latency_undefined_reason": None if paired_latency else "no matched no-fault latency",
    }
