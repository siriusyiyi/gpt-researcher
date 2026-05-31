import pytest
import numpy as np

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.compression.context_aware import ContextAwareCompressor


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()


@pytest.mark.asyncio
async def test_compressor_deduplicates_near_duplicates():
    embeddings = FakeEmbeddings()
    compressor = ContextAwareCompressor(embeddings=embeddings, dedup_threshold=0.5, max_results=5)
    # Use enough chunks to exceed max_results so the fast path is skipped
    # and the dedup logic via embeddings runs. Identical text produces
    # identical embedding vectors (same RNG seed), so cosine similarity = 1.0.
    chunks = [
        Chunk(content="identical text about topic", metadata={"source": "a.com"}, rerank_score=0.9),
        Chunk(content="identical text about topic", metadata={"source": "b.com"}, rerank_score=0.7),
        Chunk(content="identical text about topic", metadata={"source": "c.com"}, rerank_score=0.5),
        Chunk(content="identical text about topic", metadata={"source": "d.com"}, rerank_score=0.3),
        Chunk(content="identical text about topic", metadata={"source": "e.com"}, rerank_score=0.2),
        Chunk(content="identical text about topic", metadata={"source": "f.com"}, rerank_score=0.1),
    ]
    result = await compressor.compress("topic query", chunks)
    assert result.count("identical text about topic") == 1


@pytest.mark.asyncio
async def test_compressor_respects_max_results():
    embeddings = FakeEmbeddings()
    compressor = ContextAwareCompressor(embeddings=embeddings, max_results=3)
    chunks = [Chunk(content=f"unique chunk number {i}", metadata={"source": f"http://{i}.com"}) for i in range(10)]
    result = await compressor.compress("test query", chunks)
    source_count = result.count("Source:")
    assert source_count <= 3


@pytest.mark.asyncio
async def test_compressor_format_includes_source_and_content():
    embeddings = FakeEmbeddings()
    compressor = ContextAwareCompressor(embeddings=embeddings, max_results=5)
    chunks = [
        Chunk(content="Test content here.", metadata={"source": "http://test.com", "title": "Test"}),
    ]
    result = await compressor.compress("test query", chunks)
    assert "Source: http://test.com" in result
    assert "Test content here." in result


@pytest.mark.asyncio
async def test_compressor_empty_input():
    embeddings = FakeEmbeddings()
    compressor = ContextAwareCompressor(embeddings=embeddings)
    result = await compressor.compress("query", [])
    assert result == ""


@pytest.mark.asyncio
async def test_compressor_small_doc_fast_path():
    embeddings = FakeEmbeddings()
    compressor = ContextAwareCompressor(embeddings=embeddings, max_results=10)
    chunks = [
        Chunk(content="Short content.", metadata={"source": "a.com"}),
    ]
    result = await compressor.compress("query", chunks)
    assert "Short content." in result
