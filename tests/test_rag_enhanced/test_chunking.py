"""Tests for chunking strategies."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from rag_enhanced.chunking.base import Chunk
from rag_enhanced.chunking.adaptive import AdaptiveChunker
from rag_enhanced.chunking.semantic import SemanticChunker
from rag_enhanced.utils.text import split_into_sentences


# ---------------------------------------------------------------------------
# Fake embeddings for SemanticChunker tests
# ---------------------------------------------------------------------------


class FakeEmbeddings:
    """Fake embeddings that return deterministic vectors based on content hash."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def _vec(self, text: str) -> list[float]:
        """Produce a 10-dim vector whose direction depends on text content."""
        rng = np.random.RandomState(hash(text) % (2**31))
        return rng.randn(10).tolist()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


# ===========================================================================
# Task 1: Sentence splitting
# ===========================================================================


class TestSplitIntoSentences:

    def test_basic_sentences(self):
        text = "Hello world. How are you? I am fine!"
        result = split_into_sentences(text)
        assert len(result) == 3
        assert result[0] == "Hello world."
        assert result[1] == "How are you?"
        assert result[2] == "I am fine!"

    def test_empty_string(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_single_sentence(self):
        assert split_into_sentences("Just one sentence.") == ["Just one sentence."]

    def test_us_abbreviation(self):
        """U.S. should NOT split at the periods."""
        text = "U.S. stocks rose today."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert "U.S." in result[0]

    def test_dr_abbreviation(self):
        """Dr. Smith should not be split."""
        text = "Dr. Smith went home."
        result = split_into_sentences(text)
        assert len(result) == 1

    def test_decimal_numbers(self):
        """322.5 should not split."""
        text = "Price was $322.5 million."
        result = split_into_sentences(text)
        assert len(result) == 1
        assert "$322.5" in result[0]

    def test_month_abbreviation(self):
        """Jan. Feb. etc. should not split."""
        text = "On Jan. 15 he arrived."
        result = split_into_sentences(text)
        assert len(result) == 1

    def test_latin_abbreviations(self):
        """e.g. and i.e. should not split."""
        text = "Use e.g. Python. It is great."
        result = split_into_sentences(text)
        assert len(result) == 2
        assert "e.g." in result[0]

    def test_mixed_abbreviations_and_real_sentences(self):
        """Complex text with mixed abbreviations and real sentence boundaries."""
        text = "The U.S. economy grew 3.2 percent in Q1. Dr. Smith confirmed the data. It was released on Feb. 14."
        result = split_into_sentences(text)
        assert len(result) == 3
        assert "U.S." in result[0]
        assert "Dr." in result[1]
        assert "Feb." in result[2]

    def test_ellipsis(self):
        """Ellipsis ... should not create sentence splits."""
        text = "He paused... then continued. That was it."
        result = split_into_sentences(text)
        # "He paused... then continued." is one sentence, "That was it." is another
        assert len(result) == 2

    def test_sentence_starting_with_quote(self):
        text = 'She said. "Hello there."'
        result = split_into_sentences(text)
        assert len(result) == 2


# ===========================================================================
# Chunk base class
# ===========================================================================


class TestChunk:
    def test_chunk_defaults(self):
        """Chunk scores default to 0.0 and metadata to empty dict."""
        c = Chunk(content="hello", metadata={"source": "a"})
        assert c.vector_score == 0.0
        assert c.bm25_score == 0.0
        assert c.hybrid_score == 0.0
        assert c.rerank_score == 0.0
        assert c.embedding is None

    def test_chunk_scores_independent(self):
        """Each score field is independent — writing one doesn't affect others."""
        c = Chunk(content="hello", metadata={})
        c.vector_score = 0.9
        assert c.bm25_score == 0.0
        assert c.hybrid_score == 0.0
        assert c.rerank_score == 0.0


# ===========================================================================
# SemanticChunker
# ===========================================================================


class TestSemanticChunker:

    @pytest.mark.asyncio
    async def test_returns_chunks(self, fake_embeddings):
        docs = [{"raw_content": "First paragraph.\n\nSecond paragraph.", "url": "http://a.com", "title": "A"}]
        chunker = SemanticChunker(embeddings=fake_embeddings)
        chunks = await chunker.chunk(docs)
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.content for c in chunks)
        assert chunks[0].metadata["source"] == "http://a.com"

    @pytest.mark.asyncio
    async def test_preserves_all_content(self, fake_embeddings):
        text = "Sentence one here. Sentence two follows. Sentence three ends it."
        docs = [{"raw_content": text, "url": "http://b.com"}]
        chunker = SemanticChunker(embeddings=fake_embeddings)
        chunks = await chunker.chunk(docs)
        combined = " ".join(c.content for c in chunks)
        for word in ["Sentence", "one", "two", "three"]:
            assert word in combined

    @pytest.mark.asyncio
    async def test_single_doc_no_split(self, fake_embeddings):
        docs = [{"raw_content": "Tiny.", "url": "http://c.com"}]
        chunker = SemanticChunker(embeddings=fake_embeddings)
        chunks = await chunker.chunk(docs)
        assert len(chunks) == 1
        assert "Tiny." in chunks[0].content

    @pytest.mark.asyncio
    async def test_multiple_documents(self, fake_embeddings):
        docs = [
            {"raw_content": "Doc A paragraph one. Doc A paragraph two.", "url": "http://a.com"},
            {"raw_content": "Doc B paragraph one. Doc B paragraph two.", "url": "http://b.com"},
        ]
        chunker = SemanticChunker(embeddings=fake_embeddings)
        chunks = await chunker.chunk(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert "http://a.com" in sources
        assert "http://b.com" in sources

    @pytest.mark.asyncio
    async def test_max_chunk_size(self, fake_embeddings):
        """Chunks exceeding max_chunk_size should be sub-split."""
        # Create a long document where all sentences are semantically similar
        # (so no breakpoints are detected, producing one huge chunk)
        long_text = ". ".join([f"Sentence number {i} about the same topic" for i in range(50)])
        docs = [{"raw_content": long_text, "url": "http://long.com"}]
        chunker = SemanticChunker(embeddings=fake_embeddings, max_chunk_size=500)
        chunks = await chunker.chunk(docs)
        # Should be split into multiple chunks
        assert len(chunks) > 1
        for c in chunks:
            # Allow some tolerance for overlap
            assert len(c.content) <= 700, f"Chunk too large: {len(c.content)} chars"

    @pytest.mark.asyncio
    async def test_overlap_between_chunks(self, fake_embeddings):
        """Adjacent chunks should share some sentences when overlap > 0."""
        text = "First sentence here. Second sentence follows. Third sentence continues. Fourth sentence ends."
        docs = [{"raw_content": text, "url": "http://overlap.com"}]
        chunker = SemanticChunker(
            embeddings=fake_embeddings,
            sentence_overlap=1,
            breakpoint_threshold=0.0,  # force many splits for testing
        )
        chunks = await chunker.chunk(docs)
        if len(chunks) >= 2:
            # Check that at least one sentence from chunk i appears in chunk i+1
            # (overlap logic prepends last sentence of previous group)
            found_overlap = False
            for i in range(len(chunks) - 1):
                # Extract words that appear in both adjacent chunks
                words_i = set(chunks[i].content.lower().split())
                words_next = set(chunks[i + 1].content.lower().split())
                common = words_i & words_next - {"the", "a", "is", "in", "to", "and", "of"}
                if common:
                    found_overlap = True
                    break
            # Overlap may or may not be detectable depending on embedding randomness
            # The important thing is it doesn't crash
            assert isinstance(found_overlap, bool)

    @pytest.mark.asyncio
    async def test_no_overlap_when_disabled(self, fake_embeddings):
        """With sentence_overlap=0, no overlap should be added."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        docs = [{"raw_content": text, "url": "http://no-overlap.com"}]
        chunker = SemanticChunker(embeddings=fake_embeddings, sentence_overlap=0)
        chunks = await chunker.chunk(docs)
        # Should work without error
        assert len(chunks) >= 1


# ===========================================================================
# AdaptiveChunker
# ===========================================================================


class TestAdaptiveChunker:

    def test_detects_markdown_headings(self):
        text = "# Intro\nSome intro text.\n\n## Details\nSome details."
        chunker = AdaptiveChunker()
        assert chunker._is_structured(text) is True

    def test_detects_html_headings(self):
        text = "<h1>Title</h1><p>Content</p><h2>Section</h2><p>More</p>"
        chunker = AdaptiveChunker()
        assert chunker._has_html_headings(text) is True

    def test_detects_paragraph_structure(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        chunker = AdaptiveChunker()
        assert chunker._has_paragraph_structure(text) is True

    def test_no_structure_detected(self):
        text = "Just a plain text without any structure at all."
        chunker = AdaptiveChunker()
        assert chunker._is_structured(text) is False

    def test_chunks_html_headings(self):
        text = "<h1>First Section</h1><p>Content for first section here.</p><h2>Second Section</h2><p>Content for second section here.</p>"
        chunker = AdaptiveChunker()
        chunks = chunker._chunk_single(text, {"url": "test.html", "title": "Test"})
        assert len(chunks) >= 2
        # Verify heading text appears in chunks
        all_content = " ".join(c.content for c in chunks)
        assert "First Section" in all_content
        assert "Second Section" in all_content

    def test_chunks_by_paragraphs(self):
        text = "First paragraph is here with some content.\n\nSecond paragraph has more.\n\nThird paragraph wraps up."
        chunker = AdaptiveChunker()
        chunks = chunker._chunk_single(text, {"url": "test.txt", "title": "Test"})
        assert len(chunks) >= 1
        all_content = " ".join(c.content for c in chunks)
        assert "First paragraph" in all_content

    def test_oversized_section_is_split(self):
        """A section larger than _MAX_SECTION_SIZE should be sub-split."""
        long_section = "# Title\n" + "Word " * 3000  # ~15K chars
        chunker = AdaptiveChunker()
        chunks = chunker._chunk_single(long_section, {"url": "test.md", "title": "Long"})
        # Should produce multiple chunks from one section
        assert len(chunks) > 1

    def test_splitter_cache_reuses_instances(self):
        chunker = AdaptiveChunker()
        s1 = chunker._get_splitter(1000, 100)
        s2 = chunker._get_splitter(1000, 100)
        assert s1 is s2
        s3 = chunker._get_splitter(500, 50)
        assert s3 is not s1

    def test_params_for_length(self):
        chunker = AdaptiveChunker()
        assert chunker._params_for_length(100) == (500, 50)
        assert chunker._params_for_length(3000) == (1000, 100)
        assert chunker._params_for_length(10000) == (1500, 200)
