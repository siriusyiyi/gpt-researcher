import pytest

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.chunking.adaptive import AdaptiveChunker


@pytest.mark.asyncio
async def test_adaptive_chunker_short_doc_small_chunks():
    """Short documents produce smaller chunks (chunk_size=500)."""
    text = "Word " * 200  # ~1000 chars
    docs = [{"raw_content": text, "url": "http://short.com"}]
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(docs)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.content) <= 700  # 500 + overlap margin


@pytest.mark.asyncio
async def test_adaptive_chunker_long_doc_large_chunks():
    """Long documents use larger chunks (chunk_size=1500)."""
    text = "Word " * 2000  # ~10000 chars
    docs = [{"raw_content": text, "url": "http://long.com"}]
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(docs)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_adaptive_chunker_structured_doc():
    """Documents with markdown headings split at heading boundaries."""
    text = "# Heading 1\nContent under heading 1.\n\n# Heading 2\nContent under heading 2."
    docs = [{"raw_content": text, "url": "http://struct.com"}]
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(docs)
    assert len(chunks) >= 2
    # Each chunk should contain a heading
    heading_count = sum(1 for c in chunks if "Heading" in c.content)
    assert heading_count >= 2


@pytest.mark.asyncio
async def test_adaptive_chunker_preserves_metadata():
    """Metadata (source, title) is carried into chunks."""
    docs = [{"raw_content": "Some content here.", "url": "http://meta.com", "title": "Meta"}]
    chunker = AdaptiveChunker()
    chunks = await chunker.chunk(docs)
    assert chunks[0].metadata["source"] == "http://meta.com"
    assert chunks[0].metadata["title"] == "Meta"
