"""Adaptive chunking strategy — adjusts parameters per document characteristics."""

from __future__ import annotations

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker, Chunk

# Thresholds for document length classification
_SHORT_DOC_THRESHOLD = 2000   # chars
_LONG_DOC_THRESHOLD = 5000    # chars

# Max chars for a single structured section (headings, paragraphs)
_MAX_SECTION_SIZE = 2000

# Heading patterns for structured document detection
_MD_HEADING_PATTERN = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_HTML_HEADING_PATTERN = re.compile(r"<h[1-6][>\s]", re.IGNORECASE)

# Paragraph breaks (double newline)
_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


class AdaptiveChunker(BaseChunker):
    """Split documents with parameters adapted to each document's length and structure.

    Strategy selection:
        - Structured (has headings) → split at heading boundaries
        - Structured (HTML headings) → split at <h1>-<h6> boundaries
        - Structured (paragraph breaks) → split at paragraph boundaries
        - Long (>5000 chars) → chunk_size=1500, overlap=200
        - Short (<2000 chars) → chunk_size=500, overlap=50
        - Medium (default) → chunk_size=1000, overlap=100
    """

    def __init__(self, default_chunk_size: int = 1000, default_overlap: int = 100):
        self.default_chunk_size = default_chunk_size
        self.default_overlap = default_overlap
        # Cache splitters by (chunk_size, overlap) to avoid recreating
        self._splitter_cache: dict[tuple[int, int], RecursiveCharacterTextSplitter] = {}

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
        if self._has_html_headings(text):
            return self._chunk_html_headings(text, source_doc)
        if self._has_md_headings(text):
            return self._chunk_md_headings(text, source_doc)
        if self._has_paragraph_structure(text):
            return self._chunk_by_paragraphs(text, source_doc)
        chunk_size, overlap = self._params_for_length(len(text))
        return self._chunk_with_splitter(text, source_doc, chunk_size, overlap)

    # ------------------------------------------------------------------
    # Structure detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_md_headings(text: str) -> bool:
        return len(_MD_HEADING_PATTERN.findall(text)) >= 2

    @staticmethod
    def _has_html_headings(text: str) -> bool:
        return len(_HTML_HEADING_PATTERN.findall(text)) >= 2

    @staticmethod
    def _has_paragraph_structure(text: str) -> bool:
        return len(_PARAGRAPH_PATTERN.findall(text)) >= 3

    def _is_structured(self, text: str) -> bool:
        """Detect if a document has any detectable structure."""
        return (
            self._has_md_headings(text)
            or self._has_html_headings(text)
            or self._has_paragraph_structure(text)
        )

    # ------------------------------------------------------------------
    # Chunking strategies
    # ------------------------------------------------------------------

    def _chunk_md_headings(self, text: str, source_doc: dict) -> list[Chunk]:
        """Split at markdown heading boundaries, with size protection."""
        sections = re.split(r"(?=\n#{1,6}\s+)", text)
        sections = [s.strip() for s in sections if s.strip()]
        return self._sections_to_chunks(sections, source_doc)

    def _chunk_html_headings(self, text: str, source_doc: dict) -> list[Chunk]:
        """Split at HTML heading boundaries, strip tags, with size protection."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        headings = soup.find_all(re.compile(r"^h[1-6]$", re.IGNORECASE))

        if not headings:
            # Fallback: treat as unstructured
            chunk_size, overlap = self._params_for_length(len(text))
            return self._chunk_with_splitter(text, source_doc, chunk_size, overlap)

        sections: list[str] = []
        for i, heading in enumerate(headings):
            # Collect content between this heading and the next
            parts = [heading.get_text()]
            sibling = heading.next_sibling
            while sibling:
                if hasattr(sibling, "name") and sibling.name and re.match(r"^h[1-6]$", sibling.name, re.IGNORECASE):
                    break
                if hasattr(sibling, "get_text"):
                    parts.append(sibling.get_text())
                elif isinstance(sibling, str) and sibling.strip():
                    parts.append(sibling.strip())
                sibling = sibling.next_sibling
            section_text = " ".join(p for p in parts if p and p.strip())
            if section_text.strip():
                sections.append(section_text.strip())

        if not sections:
            chunk_size, overlap = self._params_for_length(len(text))
            return self._chunk_with_splitter(text, source_doc, chunk_size, overlap)

        return self._sections_to_chunks(sections, source_doc)

    def _chunk_by_paragraphs(self, text: str, source_doc: dict) -> list[Chunk]:
        """Split at paragraph boundaries, merging short paragraphs."""
        paragraphs = _PARAGRAPH_PATTERN.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Merge short paragraphs
        merged: list[str] = []
        for para in paragraphs:
            if merged and len(para) < _SHORT_DOC_THRESHOLD:
                merged[-1] += "\n\n" + para
            else:
                merged.append(para)

        return self._sections_to_chunks(merged, source_doc)

    def _sections_to_chunks(self, sections: list[str], source_doc: dict) -> list[Chunk]:
        """Convert sections to Chunks, sub-splitting oversized ones."""
        chunks: list[Chunk] = []
        idx = 0
        for section in sections:
            if len(section) > _MAX_SECTION_SIZE:
                sub = self._get_splitter(_MAX_SECTION_SIZE, 100).split_text(section)
                for s in sub:
                    chunks.append(self._make_chunk(s, source_doc, idx))
                    idx += 1
            else:
                chunks.append(self._make_chunk(section, source_doc, idx))
                idx += 1
        return chunks

    def _chunk_with_splitter(self, text: str, source_doc: dict, chunk_size: int, overlap: int) -> list[Chunk]:
        """Use RecursiveCharacterTextSplitter with the given parameters."""
        splitter = self._get_splitter(chunk_size, overlap)
        texts = splitter.split_text(text)
        return [self._make_chunk(t, source_doc, i) for i, t in enumerate(texts)]

    def _get_splitter(self, chunk_size: int, overlap: int) -> RecursiveCharacterTextSplitter:
        """Get or create a cached splitter instance."""
        key = (chunk_size, overlap)
        if key not in self._splitter_cache:
            self._splitter_cache[key] = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
        return self._splitter_cache[key]

    @staticmethod
    def _make_chunk(content: str, source_doc: dict, index: int) -> Chunk:
        return Chunk(
            content=content,
            metadata={
                "source": source_doc.get("url", ""),
                "title": source_doc.get("title", ""),
                "chunk_index": index,
            },
        )

    def _params_for_length(self, length: int) -> tuple[int, int]:
        """Return (chunk_size, overlap) based on document length."""
        if length < _SHORT_DOC_THRESHOLD:
            return 500, 50
        if length > _LONG_DOC_THRESHOLD:
            return 1500, 200
        return self.default_chunk_size, self.default_overlap
