"""Adaptive chunking strategy — adjusts parameters per document characteristics."""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker, Chunk

# Thresholds for document length classification
_SHORT_DOC_THRESHOLD = 2000   # chars
_LONG_DOC_THRESHOLD = 5000    # chars

# Heading pattern for structured document detection
_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)


class AdaptiveChunker(BaseChunker):
    """Split documents with parameters adapted to each document's length and structure.

    Strategy selection:
        - Structured (has headings) → split at heading boundaries
        - Long (>5000 chars) → chunk_size=1500, overlap=200
        - Short (<2000 chars) → chunk_size=500, overlap=50
        - Medium (default) → chunk_size=1000, overlap=100
    """

    def __init__(self, default_chunk_size: int = 1000, default_overlap: int = 100):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap

    async def chunk(self, documents: list[dict]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            raw = doc.get("raw_content", "")
            if not raw:
                continue
            chunks = self._chunk_single(raw, doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_single(self, text: str, source_doc: dict) -> list[Chunk]:
        """Choose strategy and chunk a single document."""
        if self._is_structured(text):
            return self._chunk_structured(text, source_doc)
        chunk_size, overlap = self._params_for_length(len(text))
        return self._chunk_with_splitter(text, source_doc, chunk_size, overlap)

    def _is_structured(self, text: str) -> bool:
        """Detect if a document has heading structure (markdown # headings)."""
        return len(_HEADING_PATTERN.findall(text)) >= 2

    def _params_for_length(self, length: int) -> tuple[int, int]:
        """Return (chunk_size, overlap) based on document length."""
        if length < _SHORT_DOC_THRESHOLD:
            return 500, 50
        if length > _LONG_DOC_THRESHOLD:
            return 1500, 200
        return self.default_chunk_size, self.default_overlap

    def _chunk_structured(self, text: str, source_doc: dict) -> list[Chunk]:
        """Split at heading boundaries, keeping the heading with its content."""
        sections = re.split(r"(?=\n#{1,6}\s+)", text)
        sections = [s.strip() for s in sections if s.strip()]
        return [
            Chunk(
                content=section,
                metadata={
                    "source": source_doc.get("url", ""),
                    "title": source_doc.get("title", ""),
                    "chunk_index": i,
                },
            )
            for i, section in enumerate(sections)
        ]

    def _chunk_with_splitter(self, text: str, source_doc: dict, chunk_size: int, overlap: int) -> list[Chunk]:
        """Use RecursiveCharacterTextSplitter with the given parameters."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        texts = splitter.split_text(text)
        return [
            Chunk(
                content=t,
                metadata={
                    "source": source_doc.get("url", ""),
                    "title": source_doc.get("title", ""),
                    "chunk_index": i,
                },
            )
            for i, t in enumerate(texts)
        ]
