import time
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field

from casepilot.config import StrictModel
from casepilot.rag.ingestion import load_policy
from casepilot.rag.sparse import SparseRetriever


class RetrievalSettings(StrictModel):
    embedding_model: str
    embedding_revision: str
    reranker_model: str
    reranker_revision: str
    device: str = "cpu"
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    max_rounds: int = Field(default=2, ge=1, le=2)


class RetrievalEngine:
    def __init__(self, strategy, *, manifest_path="configs/benchmark-manifest.json"):
        self.strategy = strategy
        self.settings = RetrievalSettings.model_validate(
            yaml.safe_load(Path("configs/retrieval.yaml").read_text(encoding="utf-8"))
        )
        if self.settings.device != "cpu":
            raise ValueError("The local MVP uses CPU retrieval models")
        self.chunks = load_policy(manifest_path)
        sparse = SparseRetriever(self.chunks)
        if strategy == "bm25":
            self.retriever = sparse
        else:
            from casepilot.rag.dense import DenseRetriever

            dense = DenseRetriever(
                self.chunks,
                model=self.settings.embedding_model,
                revision=self.settings.embedding_revision,
            )
            if strategy == "dense":
                self.retriever = dense
            else:
                from casepilot.rag.hybrid import HybridRetriever

                hybrid = HybridRetriever(
                    sparse, dense, candidate_k=self.settings.candidate_k, rrf_k=self.settings.rrf_k
                )
                if strategy == "hybrid":
                    self.retriever = hybrid
                elif strategy == "hybrid_rerank":
                    from casepilot.rag.reranker import Reranker

                    self.retriever = Reranker(
                        hybrid,
                        model=self.settings.reranker_model,
                        revision=self.settings.reranker_revision,
                    )
                else:
                    raise ValueError("Unknown retrieval strategy")

    def search(self, query, top_k=None, *, retrieval_round=1):
        if retrieval_round not in (1, 2):
            raise ValueError("Retrieval rounds must be 1..2")
        started = time.perf_counter()
        hits = self.retriever.search(query, top_k or self.settings.top_k)
        elapsed = (time.perf_counter() - started) * 1000
        return [
            {
                **hit.chunk.model_dump(mode="json"),
                "rank": hit.rank,
                "score": hit.score,
                "component_scores": hit.component_scores,
                "strategy": self.strategy,
                "retrieval_round": retrieval_round,
                "latency_ms": elapsed,
            }
            for hit in hits
        ]


@lru_cache(maxsize=8)
def get_engine(strategy, manifest_path="configs/benchmark-manifest.json"):
    return RetrievalEngine(strategy, manifest_path=manifest_path)
