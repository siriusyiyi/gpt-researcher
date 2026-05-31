import pytest
from rag_enhanced.chunking.base import Chunk


def test_chunk_defaults():
    """Chunk scores default to 0.0 and metadata to empty dict."""
    c = Chunk(content="hello", metadata={"source": "a"})
    assert c.vector_score == 0.0
    assert c.bm25_score == 0.0
    assert c.hybrid_score == 0.0
    assert c.rerank_score == 0.0


def test_chunk_scores_independent():
    """Each score field is independent — writing one doesn't affect others."""
    c = Chunk(content="hello", metadata={})
    c.vector_score = 0.9
    assert c.bm25_score == 0.0
    assert c.hybrid_score == 0.0
    assert c.rerank_score == 0.0
