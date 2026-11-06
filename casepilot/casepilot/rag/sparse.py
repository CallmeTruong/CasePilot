import re

from casepilot.rag.chunking import RetrievalHit


def tokenize(text):
    return re.findall(r"\w+", text.casefold())


class SparseRetriever:
    def __init__(self, chunks):
        from rank_bm25 import BM25Okapi

        if not chunks:
            raise ValueError("Cannot index an empty policy corpus")
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.section + " " + c.text) for c in chunks])

    def search(self, query, top_k=5):
        if not tokenize(query) or top_k < 1:
            return []
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].chunk_id))
        return [
            RetrievalHit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                rank=rank,
                component_scores={"bm25": float(scores[i])},
            )
            for rank, i in enumerate([i for i in order if scores[i] > 0][:top_k], 1)
        ]
