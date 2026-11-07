import hashlib

import pytest

from casepilot.rag.chunking import chunk_policy
from casepilot.rag.hybrid import fuse_rankings
from casepilot.rag.sparse import SparseRetriever


def chunks():
    return chunk_policy(
        "# Retail policy\nVerify identity.\n## Returns\nReturn delivered orders.\n"
        "## Cancellation\nCancel pending orders only.",
        source="retail/policy.md",
        version="fixture",
    )


def test_chunks_stable_grounded_without_invented_dates():
    first, second = chunks(), chunks()
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(c.text_hash == hashlib.sha256(c.text.encode()).hexdigest() for c in first)
    assert all(c.effective_from is None and c.effective_to is None for c in first)
    assert len({c.chunk_id for c in first}) == len(first)


def test_sparse_retrieves_correct_evidence_and_empty_query():
    retriever = SparseRetriever(chunks())
    hits = retriever.search("return delivered orders", 2)
    assert hits[0].chunk.section.endswith("Returns")
    assert retriever.search("", 3) == []
    assert retriever.search("zzznotincorpus", 3) == []


def test_fusion_deduplicates_and_ranks_consistent_evidence():
    retriever = SparseRetriever(chunks())
    first = retriever.search("return delivered orders", 3)
    result = fuse_rankings([first, first], top_k=3)
    assert len({h.chunk.chunk_id for h in result}) == len(result)
    assert result[0].chunk.chunk_id == first[0].chunk.chunk_id
    assert result[0].score == pytest.approx(2 / 61)
