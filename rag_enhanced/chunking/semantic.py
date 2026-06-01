"""Semantic chunking strategy — splits at embedding similarity breakpoints."""

from __future__ import annotations

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..utils.text import cosine_similarity, split_into_sentences
from .base import BaseChunker, Chunk

# Minimum characters per chunk to avoid tiny fragments
_MIN_CHUNK_CHARS = 100

# Default max chunk size (chars) to prevent oversized chunks
_DEFAULT_MAX_CHUNK_SIZE = 2000

# Default overlap in sentences between adjacent chunks
_DEFAULT_SENTENCE_OVERLAP = 1


class SemanticChunker(BaseChunker):
    """Split documents at semantic boundaries detected via embedding similarity.

    Algorithm:
        1. Pre-split each document into sentences.
        2. Batch-embed all sentences.
        3. Compute cosine similarity between adjacent sentence embeddings.
        4. Detect breakpoints where similarity drops below the rolling mean
           minus `breakpoint_threshold` * standard-deviation.
        5. Group sentences between breakpoints into chunks, merging any
           chunk shorter than min_chunk_chars with its neighbour.
        6. Add sentence-level overlap between adjacent chunks.
        7. Split any chunk exceeding max_chunk_size with a fallback splitter.
    """

    def __init__(
        self,
        embeddings,
        breakpoint_threshold: float = 0.3,
        min_chunk_chars: int = _MIN_CHUNK_CHARS,
        max_chunk_size: int = _DEFAULT_MAX_CHUNK_SIZE,
        sentence_overlap: int = _DEFAULT_SENTENCE_OVERLAP,
    ):
        self.embeddings = embeddings
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap

    async def chunk(self, documents: list[dict]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            raw = doc.get("raw_content", "")
            if not raw:
                continue
            chunks = await self._chunk_single(raw, doc)
            all_chunks.extend(chunks)
        return all_chunks

    async def _chunk_single(self, text: str, source_doc: dict) -> list[Chunk]:
        """Chunk a single document's text."""
        sentences = split_into_sentences(text)
        if len(sentences) <= 1:
            return [self._make_chunk(text, source_doc, 0)]

        # Embed all sentences
        embeddings = await self.embeddings.aembed_documents(sentences)

        # Similarities between adjacent pairs
        similarities = [
            cosine_similarity(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        if not similarities:
            return [self._make_chunk(text, source_doc, 0)]

        # Breakpoint detection: similarity < mean - threshold * std
        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))
        breakpoint_val = mean_sim - self.breakpoint_threshold * std_sim

        # Find split indices (between sentence i and i+1)
        split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < breakpoint_val]

        # Build chunks from sentence groups
        groups: list[list[str]] = []
        prev = 0
        for idx in split_indices:
            groups.append(sentences[prev:idx])
            prev = idx
        groups.append(sentences[prev:])

        # Merge tiny groups into neighbours
        merged: list[list[str]] = []
        for group in groups:
            group_text = " ".join(group)
            if merged and len(group_text) < self.min_chunk_chars:
                merged[-1].extend(group)
            else:
                merged.append(list(group))  # copy to avoid aliasing

        # Add sentence overlap between adjacent groups
        overlapped = self._add_overlap(merged)

        # Build final chunks (may need sub-splitting for oversized ones)
        chunks: list[Chunk] = []
        chunk_idx = 0
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=100,
        )
        for group in overlapped:
            group_text = " ".join(group)
            if len(group_text) > self.max_chunk_size:
                # Sub-split oversized chunk
                sub_texts = splitter.split_text(group_text)
                for st in sub_texts:
                    chunks.append(self._make_chunk(st, source_doc, chunk_idx))
                    chunk_idx += 1
            else:
                chunks.append(self._make_chunk(group_text, source_doc, chunk_idx))
                chunk_idx += 1

        return chunks

    def _add_overlap(self, groups: list[list[str]]) -> list[list[str]]:
        """Add sentence overlap between adjacent groups.

        Each group (except the first) gets the last N sentences of the
        previous group prepended. The overlap is not counted toward
        max_chunk_size limits.
        """
        if self.sentence_overlap <= 0 or len(groups) <= 1:
            return groups

        result: list[list[str]] = [list(groups[0])]
        for i in range(1, len(groups)):
            prev_tail = result[i - 1][-self.sentence_overlap:]
            result.append(prev_tail + list(groups[i]))
        return result

    def _make_chunk(self, content: str, source_doc: dict, index: int) -> Chunk:
        return Chunk(
            content=content,
            metadata={
                "source": source_doc.get("url", ""),
                "title": source_doc.get("title", ""),
                "chunk_index": index,
            },
        )
