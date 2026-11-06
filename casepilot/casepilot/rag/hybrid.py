from casepilot.rag.chunking import RetrievalHit


def fuse_rankings(rankings, *, top_k=5, rrf_k=60):
    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")
    scores, chunks, components = {}, {}, {}
    for ranking in rankings:
        seen = set()
        for hit in ranking:
            key = hit.chunk.chunk_id
            if key in seen:
                continue
            seen.add(key)
            chunks[key] = hit.chunk
            scores[key] = scores.get(key, 0) + 1 / (rrf_k + hit.rank)
            components.setdefault(key, {}).update(hit.component_scores)
    ids = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    return [
        RetrievalHit(
            chunk=chunks[key],
            score=scores[key],
            rank=rank,
            component_scores=components[key] | {"rrf": scores[key]},
        )
        for rank, key in enumerate(ids, 1)
    ]


class HybridRetriever:
    def __init__(self, sparse, dense, *, candidate_k=20, rrf_k=60):
        self.sparse, self.dense = sparse, dense
        self.candidate_k, self.rrf_k = candidate_k, rrf_k

    def search(self, query, top_k=5):
        return fuse_rankings(
            [
                self.sparse.search(query, self.candidate_k),
                self.dense.search(query, self.candidate_k),
            ],
            top_k=top_k,
            rrf_k=self.rrf_k,
        )
