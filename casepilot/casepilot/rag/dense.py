import hashlib
import json
from pathlib import Path

from casepilot.rag.chunking import RetrievalHit

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class DenseRetriever:
    def __init__(
        self, chunks, *, cache_dir=".cache/rag", model=EMBEDDING_MODEL, revision=EMBEDDING_REVISION
    ):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model_id, self.revision = model, revision
        self.encoder = SentenceTransformer(
            model, revision=revision, device="cpu", trust_remote_code=False
        )
        metadata = {
            "model": model,
            "revision": revision,
            "chunks": [(c.chunk_id, c.text_hash) for c in chunks],
        }
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        folder = Path(cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.faiss"
        if path.exists():
            self.index = faiss.read_index(str(path))
            if self.index.ntotal != len(chunks):
                raise ValueError("Dense index metadata mismatch")
        else:
            vectors = self.encoder.encode(
                [c.section + "\n" + c.text for c in chunks],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors.astype("float32"))
            faiss.write_index(self.index, str(path))
        (folder / f"{key}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def search(self, query, top_k=5):
        if not query.strip() or top_k < 1:
            return []
        vector = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, indices = self.index.search(vector.astype("float32"), min(top_k, len(self.chunks)))
        return [
            RetrievalHit(
                chunk=self.chunks[int(index)],
                score=float(score),
                rank=rank,
                component_scores={"dense": float(score)},
            )
            for rank, (score, index) in enumerate(zip(scores[0], indices[0], strict=True), 1)
        ]
