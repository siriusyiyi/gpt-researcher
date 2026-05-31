import pytest
import numpy as np
from unittest.mock import MagicMock

from rag_enhanced.adapter import RAGAdapter
from rag_enhanced.config import RAGConfig


class FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(8).tolist()


def _make_fake_researcher():
    researcher = MagicMock()
    cfg = MagicMock()
    cfg.fast_llm_model = "openai:gpt-4o-mini"
    cfg.fast_llm_provider = "openai"
    cfg.max_search_results_per_query = 10
    researcher.cfg = cfg

    memory = MagicMock()
    memory.get_embeddings.return_value = FakeEmbeddings()
    researcher.memory = memory

    researcher.verbose = False
    researcher.websocket = None
    researcher.prompt_family = MagicMock()
    researcher.kwargs = {}
    return researcher


@pytest.mark.asyncio
async def test_adapter_get_similar_content_by_query():
    researcher = _make_fake_researcher()
    config = RAGConfig(
        enable_query_rewrite=False,
        chunking_strategy="adaptive",
        reranker="embedding",
        max_results=5,
        llm_model="openai:gpt-4o-mini",
        llm_provider="openai",
        rerank_top_k=10,
    )
    adapter = RAGAdapter(researcher, config=config)

    pages = [
        {"raw_content": "Python is a programming language.", "url": "http://a.com", "title": "Python"},
        {"raw_content": "Java is also a programming language.", "url": "http://b.com", "title": "Java"},
    ]
    result = await adapter.get_similar_content_by_query("What is Python?", pages)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_adapter_config_from_researcher():
    researcher = _make_fake_researcher()
    adapter = RAGAdapter(researcher)
    assert adapter.pipeline.config.llm_model == "openai:gpt-4o-mini"
    assert adapter.pipeline.config.llm_provider == "openai"
    assert adapter.pipeline.config.rerank_top_k == 10


@pytest.mark.asyncio
async def test_adapter_empty_pages():
    researcher = _make_fake_researcher()
    adapter = RAGAdapter(researcher)
    result = await adapter.get_similar_content_by_query("query", [])
    assert result == ""


@pytest.mark.asyncio
async def test_adapter_custom_config():
    researcher = _make_fake_researcher()
    custom_config = RAGConfig(
        chunking_strategy="adaptive",
        enable_query_rewrite=False,
        rerank_top_k=5,
    )
    adapter = RAGAdapter(researcher, config=custom_config)
    assert adapter.pipeline.config.rerank_top_k == 5
    assert adapter.pipeline.config.enable_query_rewrite is False
