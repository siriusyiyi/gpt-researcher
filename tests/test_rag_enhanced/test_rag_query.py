"""Tests for the RAG query tool and convenience functions."""

import sys
import tempfile

import numpy as np
import pytest
import pytest_asyncio

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.knowledge_store.chroma_store import ChromaKnowledgeStore
from rag_enhanced.tools.rag_query import RAGQueryTool, knowledge_search


class FakeEmbeddings:
    """Minimal embeddings mock for testing."""

    def __init__(self):
        self._cache: dict[str, list[float]] = {}

    def _vec(self, text: str) -> list[float]:
        if text not in self._cache:
            rng = np.random.RandomState(hash(text) % (2**31))
            self._cache[text] = rng.randn(8).tolist()
        return self._cache[text]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _make_chunks(n: int, prefix: str = "doc", source: str = "test.txt") -> list[Chunk]:
    return [
        Chunk(
            content=f"{prefix} content {i} about Python programming and data science",
            metadata={"source": source, "title": f"Doc {i}", "chunk_index": i},
        )
        for i in range(n)
    ]


@pytest.fixture
def temp_dir():
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest_asyncio.fixture
async def populated_store(embeddings, temp_dir):
    """A knowledge store with some test data."""
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="rag_test",
        persist_directory=temp_dir,
    )
    chunks = _make_chunks(10, prefix="machine learning")
    await store.add_documents(chunks)
    return store


def _clear_store_cache():
    """Clear the module-level store cache."""
    mod = sys.modules.get("rag_enhanced.tools.rag_query")
    if mod and hasattr(mod, "_store_cache"):
        mod._store_cache.clear()


# --- RAGQueryTool class tests ---

@pytest.mark.asyncio
async def test_rag_query_tool_returns_structure(embeddings, populated_store):
    tool = RAGQueryTool(
        knowledge_store=populated_store,
        embeddings=embeddings,
    )
    result = await tool.query("machine learning")

    assert "answer" in result
    assert "sources" in result
    assert "chunks" in result
    assert "context" in result
    # No LLM → empty answer
    assert result["answer"] == ""
    # Should have found some sources
    assert len(result["sources"]) > 0
    assert len(result["chunks"]) > 0


@pytest.mark.asyncio
async def test_rag_query_tool_with_llm(embeddings, populated_store):
    async def fake_llm(messages, **kwargs):
        return "This is a test answer about machine learning."

    tool = RAGQueryTool(
        knowledge_store=populated_store,
        embeddings=embeddings,
        llm_func=fake_llm,
    )
    result = await tool.query("machine learning")

    assert result["answer"] == "This is a test answer about machine learning."
    assert len(result["sources"]) > 0


@pytest.mark.asyncio
async def test_rag_query_tool_empty_store(embeddings, temp_dir):
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="empty_test",
        persist_directory=temp_dir,
    )
    tool = RAGQueryTool(
        knowledge_store=store,
        embeddings=embeddings,
    )
    result = await tool.query("nonexistent query")

    assert result["answer"] == ""
    assert result["sources"] == []
    assert result["chunks"] == []
    assert result["context"] == ""


@pytest.mark.asyncio
async def test_rag_query_tool_search(embeddings, populated_store):
    tool = RAGQueryTool(
        knowledge_store=populated_store,
        embeddings=embeddings,
    )
    results = await tool.search("Python", top_k=3)

    assert len(results) <= 3
    for r in results:
        assert "content" in r
        assert "source" in r
        assert "score" in r


# --- Convenience function tests ---

@pytest.mark.asyncio
async def test_knowledge_search_convenience(embeddings, temp_dir):
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="conv_test",
        persist_directory=temp_dir,
    )
    chunks = _make_chunks(5, prefix="deep learning", source="ai.txt")
    await store.add_documents(chunks)

    _clear_store_cache()

    results = await knowledge_search(
        "deep learning",
        top_k=3,
        store_path=temp_dir,
        collection="conv_test",
        embeddings=embeddings,
    )

    assert len(results) <= 3
    assert all("content" in r for r in results)


@pytest.mark.asyncio
async def test_knowledge_add_convenience(embeddings, temp_dir):
    _clear_store_cache()

    from rag_enhanced.tools.rag_query import knowledge_add

    documents = [
        {"raw_content": "This is a test document about AI.", "url": "test.txt"},
        {"raw_content": "Another document about machine learning.", "url": "test2.txt"},
    ]

    count = await knowledge_add(
        documents,
        store_path=temp_dir,
        collection="add_test",
        embeddings=embeddings,
    )

    assert count > 0

    # Verify we can retrieve
    results = await knowledge_search(
        "AI",
        top_k=5,
        store_path=temp_dir,
        collection="add_test",
        embeddings=embeddings,
    )
    assert len(results) > 0
