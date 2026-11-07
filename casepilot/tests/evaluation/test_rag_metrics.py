import pytest

from casepilot.evaluation.rag_metrics import (
    citation_accuracy,
    policy_decision_accuracy,
    query_metrics,
)


def test_recall_rank_and_ndcg_against_hand_computation():
    result = query_metrics(["b", "a", "c"], {"a", "c"}, 3)
    assert result["recall"] == 1.0
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx(
        (1 / 1.584962500721156 + 0.5) / (1 + 1 / 1.584962500721156)
    )


def test_missing_gold_is_undefined_and_duplicate_hits_do_not_inflate_recall():
    assert query_metrics(["a"], set(), 5)["recall"] is None
    assert query_metrics(["a", "a", "a"], {"a", "b"}, 5)["recall"] == 0.5


def test_citation_judgments_are_not_inferred_from_keyword_overlap():
    result = citation_accuracy(
        ["citation-a", "citation-b", "unjudged"], {"citation-a": True, "citation-b": False}
    )
    assert result["accuracy"].value == 0.5 and result["unjudged"] == 1
    assert policy_decision_accuracy({"x": "deny"}, {"x": "allow"}).value == 0
    assert citation_accuracy(["a"], {})["accuracy"].value is None
