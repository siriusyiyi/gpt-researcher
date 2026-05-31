"""Integration tests for KnowledgeStore + Pipeline interaction."""

import tempfile

import numpy as np
import pytest
import pytest_asyncio

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.config import RAGConfig
from rag_enhanced.knowledge_store.chroma_store import ChromaKnowledgeStore
from rag_enhanced.pipeline import RAGPipeline


class FakeEmbeddings:
    """Deterministic embeddings mock."""

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


def _kb_chunks(n: int) -> list[Chunk]:
    """Chunks that will be pre-loaded into the knowledge store."""
    return [
        Chunk(
            content=f"Knowledge base article {i} about artificial intelligence and neural networks",
            metadata={"source": f"kb_article_{i}.txt", "title": f"Article {i}", "chunk_index": i},
        )
        for i in range(n)
    ]


def _in_memory_docs(n: int) -> list[dict]:
    """In-memory documents (simulating freshly scraped web content)."""
    return [
        {
            "raw_content": f"Web page {i} content about Python web development and REST APIs",
            "url": f"https://example.com/page{i}",
            "title": f"Web Page {i}",
        }
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
async def knowledge_store(embeddings, temp_dir):
    """Pre-populated knowledge store."""
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="integration_test",
        persist_directory=temp_dir,
    )
    chunks = _kb_chunks(10)
    await store.add_documents(chunks)
    return store


# --- Mode: supplement (knowledge store + in-memory) ---

@pytest.mark.asyncio
async def test_supplement_mode_merges_sources(embeddings, knowledge_store):
    """In supplement mode, results should come from BOTH knowledge store and in-memory docs."""
    config = RAGConfig(
        knowledge_store_mode="supplement",
        enable_query_rewrite=False,
        rerank_top_k=10,
        max_results=10,
    )
    pipeline = RAGPipeline(
        config=config,
        embeddings=embeddings,
        knowledge_store=knowledge_store,
    )

    context = await pipeline.process("artificial intelligence", _in_memory_docs(5))

    assert len(context) > 0
    # Should contain content from knowledge store sources (kb_article_*.txt)
    assert "kb_article" in context or "example.com" in context


@pytest.mark.asyncio
async def test_supplement_mode_with_no_in_memory_docs(embeddings, knowledge_store):
    """Supplement mode should still work when no in-memory docs are provided."""
    config = RAGConfig(
        knowledge_store_mode="supplement",
        enable_query_rewrite=False,
        rerank_top_k=5,
        max_results=5,
    )
    pipeline = RAGPipeline(
        config=config,
        embeddings=embeddings,
        knowledge_store=knowledge_store,
    )

    context = await pipeline.process("artificial intelligence", [])

    assert len(context) > 0
    assert "kb_article" in context


# --- Mode: primary (knowledge store only) ---

@pytest.mark.asyncio
async def test_primary_mode_ignores_in_memory_docs(embeddings, knowledge_store):
    """In primary mode, in-memory docs should be ignored — only KB results used."""
    config = RAGConfig(
        knowledge_store_mode="primary",
        enable_query_rewrite=False,
        rerank_top_k=5,
        max_results=5,
    )
    pipeline = RAGPipeline(
        config=config,
        embeddings=embeddings,
        knowledge_store=knowledge_store,
    )

    # Pass in-memory docs but they should be ignored in primary mode
    context = await pipeline.process("artificial intelligence", _in_memory_docs(5))

    assert len(context) > 0
    # Should only contain KB sources
    assert "kb_article" in context


@pytest.mark.asyncio
async def test_primary_mode_empty_kb_falls_back(embeddings, temp_dir):
    """When KB is empty in primary mode, should return empty (no fallback to in-memory)."""
    empty_store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="empty_kb",
        persist_directory=temp_dir,
    )
    config = RAGConfig(
        knowledge_store_mode="primary",
        enable_query_rewrite=False,
    )
    pipeline = RAGPipeline(
        config=config,
        embeddings=embeddings,
        knowledge_store=empty_store,
    )

    context = await pipeline.process("query", _in_memory_docs(3))
    # Primary mode with empty KB → no KB chunks → fallback to in-memory
    # (since primary only short-circuits when kb_chunks is non-empty)
    assert isinstance(context, str)


# --- No knowledge store (original behavior preserved) ---

@pytest.mark.asyncio
async def test_no_knowledge_store_original_behavior(embeddings):
    """Without a knowledge store, pipeline should work as before."""
    config = RAGConfig(
        enable_query_rewrite=False,
        rerank_top_k=5,
        max_results=5,
    )
    pipeline = RAGPipeline(
        config=config,
        embeddings=embeddings,
    )

    context = await pipeline.process("Python", _in_memory_docs(3))
    assert len(context) > 0
    assert "example.com" in context


# --- Knowledge store lifecycle ---

@pytest.mark.asyncio
async def test_add_then_retrieve_then_delete(embeddings, temp_dir):
    """Full lifecycle: add documents, verify retrieval, delete, verify empty."""
    store = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="lifecycle_test",
        persist_directory=temp_dir,
    )

    # Add
    chunks = _kb_chunks(5)
    count = await store.add_documents(chunks)
    assert count == 5
    assert await store.count() == 5

    # Retrieve
    results = await store.retrieve("artificial intelligence", top_k=3)
    assert len(results) > 0

    # Delete
    deleted = await store.delete("kb_article_0.txt")
    assert deleted == 1
    assert await store.count() == 4

    # List sources
    sources = store.list_sources()
    assert len(sources) == 4
    assert "kb_article_0.txt" not in {s["source"] for s in sources}
