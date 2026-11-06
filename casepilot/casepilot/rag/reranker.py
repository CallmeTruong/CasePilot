from casepilot.rag.chunking import RetrievalHit

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class Reranker:
    def __init__(self, retriever, *, model=RERANK_MODEL, revision=RERANK_REVISION):
        from sentence_transformers import CrossEncoder

        self.retriever = retriever
        self.model = CrossEncoder(model, revision=revision, device="cpu", trust_remote_code=False)

    def search(self, query, top_k=5):
        hits = self.retriever.search(query, max(20, top_k))
        if not hits:
            return []
        scores = self.model.predict([(query, hit.chunk.text) for hit in hits])
        ranked = sorted(
            zip(hits, scores, strict=True), key=lambda item: (-item[1], item[0].chunk.chunk_id)
        )
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=float(score),
                rank=rank,
                component_scores=hit.component_scores | {"rerank": float(score)},
            )
            for rank, (hit, score) in enumerate(ranked[:top_k], 1)
        ]
