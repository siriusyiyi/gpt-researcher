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


import numpy as np
from unittest.mock import AsyncMock, MagicMock

from rag_enhanced.chunking.semantic import SemanticChunker


class FakeEmbeddings:
    """Fake embeddings that return deterministic vectors based on content hash."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        """Produce a 10-dim vector whose direction depends on text content."""
        rng = np.random.RandomState(hash(text) % (2**31))
        v = rng.randn(10).tolist()
        return v


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


@pytest.mark.asyncio
async def test_semantic_chunker_returns_chunks(fake_embeddings):
    """SemanticChunker splits a document and returns Chunk objects."""
    docs = [{"raw_content": "First paragraph.\n\nSecond paragraph.", "url": "http://a.com", "title": "A"}]
    chunker = SemanticChunker(embeddings=fake_embeddings)
    chunks = await chunker.chunk(docs)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.content for c in chunks)
    assert chunks[0].metadata["source"] == "http://a.com"
    assert chunks[0].metadata["title"] == "A"


@pytest.mark.asyncio
async def test_semantic_chunker_preserves_all_content(fake_embeddings):
    """All original content must appear in the concatenated chunks."""
    text = "Sentence one here. Sentence two follows. Sentence three ends it."
    docs = [{"raw_content": text, "url": "http://b.com"}]
    chunker = SemanticChunker(embeddings=fake_embeddings)
    chunks = await chunker.chunk(docs)
    combined = " ".join(c.content for c in chunks)
    for word in ["Sentence", "one", "two", "three"]:
        assert word in combined


@pytest.mark.asyncio
async def test_semantic_chunker_single_doc_no_split(fake_embeddings):
    """Very short documents should produce at least one chunk."""
    docs = [{"raw_content": "Tiny.", "url": "http://c.com"}]
    chunker = SemanticChunker(embeddings=fake_embeddings)
    chunks = await chunker.chunk(docs)
    assert len(chunks) == 1
    assert "Tiny." in chunks[0].content


@pytest.mark.asyncio
async def test_semantic_chunker_multiple_documents(fake_embeddings):
    """Each input document produces its own set of chunks."""
    docs = [
        {"raw_content": "Doc A paragraph one. Doc A paragraph two.", "url": "http://a.com"},
        {"raw_content": "Doc B paragraph one. Doc B paragraph two.", "url": "http://b.com"},
    ]
    chunker = SemanticChunker(embeddings=fake_embeddings)
    chunks = await chunker.chunk(docs)
    sources = {c.metadata["source"] for c in chunks}
    assert "http://a.com" in sources
    assert "http://b.com" in sources
