import pytest
import numpy as np

from rag_enhanced.config import RAGConfig
from rag_enhanced.pipeline import RAGPipeline


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()


@pytest.mark.asyncio
async def test_pipeline_end_to_end_returns_string():
    config = RAGConfig(
        chunking_strategy="adaptive",
        enable_query_rewrite=False,
        hybrid_search=True,
        fusion_mode="rrf",
        reranker="embedding",
        max_results=5,
    )
    embeddings = FakeEmbeddings()
    pipeline = RAGPipeline(config=config, embeddings=embeddings)

    docs = [
        {"raw_content": "Python is a programming language. It is widely used.", "url": "http://a.com", "title": "Python"},
        {"raw_content": "Machine learning uses algorithms to learn from data.", "url": "http://b.com", "title": "ML"},
    ]
    result = await pipeline.process("What is Python?", docs)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Python" in result


@pytest.mark.asyncio
async def test_pipeline_with_query_rewrite():
    async def fake_llm(messages, **kwargs):
        return '- "What programming languages exist"\n- "Python language overview"'

    config = RAGConfig(
        chunking_strategy="adaptive",
        enable_query_rewrite=True,
        query_rewrite_strategy="multi",
        hybrid_search=True,
        reranker="embedding",
        max_results=5,
    )
    embeddings = FakeEmbeddings()
    pipeline = RAGPipeline(config=config, embeddings=embeddings, llm_func=fake_llm)

    docs = [
        {"raw_content": "Python is a programming language.", "url": "http://a.com"},
    ]
    result = await pipeline.process("What is Python?", docs)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_pipeline_empty_documents():
    config = RAGConfig(enable_query_rewrite=False)
    embeddings = FakeEmbeddings()
    pipeline = RAGPipeline(config=config, embeddings=embeddings)
    result = await pipeline.process("test query", [])
    assert result == ""


@pytest.mark.asyncio
async def test_pipeline_semantic_chunking():
    config = RAGConfig(
        chunking_strategy="semantic",
        enable_query_rewrite=False,
        hybrid_search=True,
        reranker="embedding",
        max_results=5,
    )
    embeddings = FakeEmbeddings()
    pipeline = RAGPipeline(config=config, embeddings=embeddings)

    docs = [
        {"raw_content": "Topic A content. " * 50 + "Topic B content. " * 50, "url": "http://a.com"},
    ]
    result = await pipeline.process("topic query", docs)
    assert isinstance(result, str)
    assert len(result) > 0
