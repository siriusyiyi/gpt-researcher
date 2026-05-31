"""Tests for the knowledge store module."""

import os
import tempfile

import numpy as np
import pytest

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.knowledge_store.chroma_store import ChromaKnowledgeStore


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


def _make_chunks(n: int, prefix: str = "doc", source: str = "test_source") -> list[Chunk]:
    return [
        Chunk(
            content=f"{prefix} content {i} about Python programming",
            metadata={"source": source, "title": f"Doc {i}", "chunk_index": i},
        )
        for i in range(n)
    ]


@pytest.fixture
def temp_store_dir():
    """Provide a temporary directory for Chroma persistence.

    Uses ignore_cleanup_errors=True because Chroma holds file locks
    on Windows that prevent temp directory deletion during teardown.
    """
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Best-effort cleanup; Chroma file locks on Windows may prevent deletion
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def embeddings():
    return FakeEmbeddings()


@pytest.fixture
def store(embeddings, temp_store_dir):
    return ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="test_collection",
        persist_directory=temp_store_dir,
    )


@pytest.mark.asyncio
async def test_add_and_count(store):
    chunks = _make_chunks(5)
    count = await store.add_documents(chunks)
    assert count == 5

    total = await store.count()
    assert total == 5


@pytest.mark.asyncio
async def test_add_empty_chunks(store):
    count = await store.add_documents([])
    assert count == 0


@pytest.mark.asyncio
async def test_retrieve_returns_chunks(store):
    chunks = _make_chunks(10, prefix="machine learning")
    await store.add_documents(chunks)

    results = await store.retrieve("machine learning", top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, Chunk) for r in results)
    # All results should have vector_score populated
    for r in results:
        assert isinstance(r.vector_score, float)


@pytest.mark.asyncio
async def test_retrieve_empty_store(store):
    results = await store.retrieve("nonexistent", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_delete_by_source(store):
    chunks_a = _make_chunks(3, prefix="alpha", source="file_a.txt")
    chunks_b = _make_chunks(3, prefix="beta", source="file_b.txt")
    await store.add_documents(chunks_a)
    await store.add_documents(chunks_b)

    assert await store.count() == 6

    deleted = await store.delete("file_a.txt")
    assert deleted == 3

    assert await store.count() == 3


@pytest.mark.asyncio
async def test_delete_nonexistent_source(store):
    chunks = _make_chunks(3)
    await store.add_documents(chunks)

    deleted = await store.delete("nonexistent.txt")
    assert deleted == 0


@pytest.mark.asyncio
async def test_list_sources(store):
    chunks_a = _make_chunks(2, source="file_a.txt")
    chunks_b = _make_chunks(3, source="file_b.txt")
    await store.add_documents(chunks_a, doc_type="local")
    await store.add_documents(chunks_b, doc_type="manual")

    sources = store.list_sources()
    assert len(sources) == 2

    source_names = {s["source"] for s in sources}
    assert source_names == {"file_a.txt", "file_b.txt"}

    # Check chunk counts
    source_map = {s["source"]: s["chunk_count"] for s in sources}
    assert source_map["file_a.txt"] == 2
    assert source_map["file_b.txt"] == 3


@pytest.mark.asyncio
async def test_add_with_doc_type_metadata(store):
    chunks = _make_chunks(2, source="web_page.html")
    await store.add_documents(chunks, doc_type="manual")

    # Verify doc_type is stored in metadata
    results = await store.retrieve("Python", top_k=2)
    assert len(results) > 0
    for r in results:
        if r.metadata.get("source") == "web_page.html":
            assert r.metadata.get("doc_type") == "manual"


@pytest.mark.asyncio
async def test_persistence(embeddings, temp_store_dir):
    """Test that data persists across store instances."""
    store1 = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="persist_test",
        persist_directory=temp_store_dir,
    )
    chunks = _make_chunks(3, prefix="persistent content")
    await store1.add_documents(chunks)

    # Create a new store instance pointing to the same directory
    store2 = ChromaKnowledgeStore(
        embeddings=embeddings,
        collection_name="persist_test",
        persist_directory=temp_store_dir,
    )
    count = await store2.count()
    assert count == 3
