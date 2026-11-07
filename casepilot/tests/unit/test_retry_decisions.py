import pytest

from casepilot.reliability.retry import recovery_decision


@pytest.mark.parametrize("strategy", ["S0", "S1", "S2"])
def test_unknown_mutation_never_blindly_retries(strategy):
    assert (
        recovery_decision(
            strategy, read_only=False, state="unknown", proof=None, attempt=1, max_retries=2
        )
        == "stop"
    )


def test_committed_receipt_reused_only_by_verifying_strategy():
    assert (
        recovery_decision(
            "S2", read_only=False, state="committed", proof="receipt", attempt=1, max_retries=2
        )
        == "receipt"
    )
    assert (
        recovery_decision(
            "S1", read_only=False, state="committed", proof="receipt", attempt=1, max_retries=2
        )
        == "stop"
    )


def test_retry_bounds_and_proven_uncommitted():
    assert (
        recovery_decision(
            "S1", read_only=False, state="failed", proof="not_dispatched", attempt=1, max_retries=2
        )
        == "retry"
    )
    assert (
        recovery_decision(
            "S2", read_only=True, state="unknown", proof=None, attempt=3, max_retries=2
        )
        == "stop"
    )
    assert (
        recovery_decision(
            "S0", read_only=True, state="failed", proof="not_dispatched", attempt=1, max_retries=2
        )
        == "stop"
    )
