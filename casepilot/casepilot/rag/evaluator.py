import time

from casepilot.evaluation.rag_metrics import query_metrics


def evaluate_retriever(retriever, queries, *, corpus_size=5):
    results = []
    for query in queries:
        started = time.perf_counter()
        hits = retriever.search(query["query"], max(5, corpus_size))
        latency = (time.perf_counter() - started) * 1000
        ids = [hit.chunk.chunk_id for hit in hits]
        results.append(
            {
                "query_id": query["query_id"],
                "ranked_ids": ids,
                "gold_ids": query["gold_ids"],
                "latency_ms": latency,
                "at3": query_metrics(ids, query["gold_ids"], 3),
                "at5": query_metrics(ids, query["gold_ids"], 5),
            }
        )
    return results
