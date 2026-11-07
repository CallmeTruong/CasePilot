"""Evaluate policy retrieval locally; no LLM/provider requests or task ground truth."""

import argparse
import json
from pathlib import Path

from casepilot.rag.engine import get_engine
from casepilot.rag.evaluator import evaluate_retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/rag_eval/queries.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("configs/benchmark-manifest.json"))
    parser.add_argument(
        "--strategies", nargs="+", default=["bm25", "dense", "hybrid", "hybrid_rerank"]
    )
    parser.add_argument("--output", type=Path, default=Path("results/E2-component.json"))
    args = parser.parse_args()
    queries = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = {}
    for strategy in args.strategies:
        engine = get_engine(strategy, str(args.manifest))
        valid_ids = {chunk.chunk_id for chunk in engine.chunks}
        if any(not set(q["gold_ids"]) <= valid_ids for q in queries):
            raise ValueError("Gold evidence IDs do not match pinned corpus")
        rows = evaluate_retriever(engine.retriever, queries, corpus_size=len(engine.chunks))
        valid = [r for r in rows if r["at5"]["recall"] is not None]

        def mean(at, metric, cohort=valid):
            return sum(r[at][metric] for r in cohort) / len(cohort) if cohort else None

        results[strategy] = {
            "queries": rows,
            "query_count": len(rows),
            "excluded_gold_empty_queries": len(rows) - len(valid),
            "recall_at_3": mean("at3", "recall"),
            "recall_at_5": mean("at5", "recall"),
            "mrr": mean("at5", "mrr"),
            "ndcg_at_5": mean("at5", "ndcg"),
            "citation_accuracy": None,
            "policy_decision_accuracy": None,
            "adjudication_status": "No independent Agent citation/decision labels yet",
        }
        print(f"{strategy}: {len(rows)} queries evaluated", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
