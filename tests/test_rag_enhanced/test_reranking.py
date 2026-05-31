import pytest
import numpy as np

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.reranking.embedding_rerank import EmbeddingReranker
from rag_enhanced.reranking.cross_encoder import CrossEncoderReranker


class FakeEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()


class FakeCrossEncoder:
    def predict(self, pairs: list[list[str]]) -> list[float]:
        return [float(len(q) + len(d)) / 100.0 for q, d in pairs]


# --- EmbeddingReranker tests ---

@pytest.mark.asyncio
async def test_embedding_reranker_populates_rerank_score():
    embeddings = FakeEmbeddings()
    reranker = EmbeddingReranker(embeddings=embeddings, top_k=5)
    chunks = [Chunk(content=f"chunk {i}", metadata={"chunk_index": i}) for i in range(10)]
    results = await reranker.rerank("test query", chunks)
    for r in results:
        assert isinstance(r.rerank_score, float)
        # Cosine similarity from random embeddings can be negative
        assert -1.0 <= r.rerank_score <= 1.0


@pytest.mark.asyncio
async def test_embedding_reranker_respects_top_k():
    embeddings = FakeEmbeddings()
    reranker = EmbeddingReranker(embeddings=embeddings, top_k=3)
    chunks = [Chunk(content=f"chunk {i}", metadata={}) for i in range(10)]
    results = await reranker.rerank("test query", chunks)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_embedding_reranker_sorted_descending():
    embeddings = FakeEmbeddings()
    reranker = EmbeddingReranker(embeddings=embeddings, top_k=10)
    chunks = [Chunk(content=f"unique chunk number {i}", metadata={}) for i in range(10)]
    results = await reranker.rerank("test query", chunks)
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_embedding_reranker_empty_input():
    embeddings = FakeEmbeddings()
    reranker = EmbeddingReranker(embeddings=embeddings, top_k=5)
    results = await reranker.rerank("test query", [])
    assert results == []


# --- CrossEncoderReranker tests ---

@pytest.mark.asyncio
async def test_cross_encoder_reranker_populates_rerank_score():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder(), top_k=3)
    chunks = [Chunk(content=f"chunk content {i}", metadata={}) for i in range(5)]
    results = await reranker.rerank("test query", chunks)
    assert len(results) == 3
    for r in results:
        assert r.rerank_score > 0.0


@pytest.mark.asyncio
async def test_cross_encoder_reranker_sorted_descending():
    fake_model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=fake_model, top_k=5)
    chunks = [
        Chunk(content="short", metadata={}),
        Chunk(content="medium length content here", metadata={}),
        Chunk(content="this is a much longer chunk with lots of content to score higher", metadata={}),
    ]
    results = await reranker.rerank("test query", chunks)
    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)
