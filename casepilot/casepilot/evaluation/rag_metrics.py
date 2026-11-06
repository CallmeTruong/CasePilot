import math

from casepilot.evaluation.metrics import ratio


def query_metrics(ranked_ids, gold_ids, k=5):
    if k < 1:
        raise ValueError("k must be positive")
    gold = set(gold_ids)
    ranked = list(dict.fromkeys(ranked_ids))
    if not gold:
        return {
            "recall": None,
            "mrr": None,
            "ndcg": None,
            "undefined_reason": "query has no gold evidence",
        }
    recall = len(set(ranked[:k]) & gold) / len(gold)
    first = next((i for i, identifier in enumerate(ranked, 1) if identifier in gold), None)
    dcg = sum(
        1 / math.log2(i + 1) for i, identifier in enumerate(ranked[:k], 1) if identifier in gold
    )
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return {
        "recall": recall,
        "mrr": 1 / first if first else 0.0,
        "ndcg": dcg / ideal,
        "undefined_reason": None,
    }


def evidence_coverage(query_results, k=5):
    gold = set().union(*(set(row["gold_ids"]) for row in query_results)) if query_results else set()
    covered = (
        set().union(*(set(row["ranked_ids"][:k]) & set(row["gold_ids"]) for row in query_results))
        if query_results
        else set()
    )
    return ratio(len(covered), len(gold), undefined_reason="no gold evidence sections")


def citation_accuracy(citation_ids, judgments):
    if len(set(citation_ids)) != len(citation_ids):
        raise ValueError("Citation IDs must uniquely identify decision/evidence links")
    judged = [
        judgments[identifier]
        for identifier in citation_ids
        if isinstance(judgments.get(identifier), bool)
    ]
    return {
        "accuracy": ratio(sum(judged), len(judged), undefined_reason="no adjudicated citations"),
        "adjudicated": len(judged),
        "unjudged": len(citation_ids) - len(judged),
    }


def policy_decision_accuracy(predictions, gold_labels):
    overlap = predictions.keys() & gold_labels.keys()
    return ratio(
        sum(predictions[key] == gold_labels[key] for key in overlap),
        len(overlap),
        undefined_reason="no gold policy decisions",
    )
