import pytest
import numpy as np

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.retrieval.hybrid import HybridRetriever


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()


def _make_chunks(n: int, content_prefix: str = "chunk") -> list[Chunk]:
    return [
        Chunk(content=f"{content_prefix} {i} with some keywords", metadata={"source": f"http://{i}.com", "chunk_index": i})
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_chunks():
    embeddings = FakeEmbeddings()
    retriever = HybridRetriever(embeddings=embeddings, top_k=5)
    chunks = _make_chunks(10)
    results = await retriever.retrieve("test query", chunks)
    assert len(results) <= 5
    assert all(isinstance(r, Chunk) for r in results)


@pytest.mark.asyncio
async def test_hybrid_retriever_rrf_populates_scores():
    embeddings = FakeEmbeddings()
    retriever = HybridRetriever(embeddings=embeddings, top_k=5, fusion_mode="rrf")
    chunks = _make_chunks(10)
    results = await retriever.retrieve("test query", chunks)
    for r in results:
        assert r.hybrid_score > 0.0
        assert r.vector_score > 0.0 or r.bm25_score > 0.0


@pytest.mark.asyncio
async def test_hybrid_retriever_weighted_fusion():
    embeddings = FakeEmbeddings()
    retriever = HybridRetriever(
        embeddings=embeddings, top_k=5,
        fusion_mode="weighted", bm25_weight=0.7, vector_weight=0.3,
    )
    chunks = _make_chunks(10)
    results = await retriever.retrieve("test query", chunks)
    assert len(results) <= 5
    for r in results:
        assert r.hybrid_score > 0.0


@pytest.mark.asyncio
async def test_hybrid_retriever_empty_chunks():
    embeddings = FakeEmbeddings()
    retriever = HybridRetriever(embeddings=embeddings, top_k=5)
    results = await retriever.retrieve("test query", [])
    assert results == []
