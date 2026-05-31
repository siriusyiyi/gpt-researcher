"""Semantic chunking strategy — splits at embedding similarity breakpoints."""

from __future__ import annotations

import numpy as np

from ..utils.text import cosine_similarity, split_into_sentences
from .base import BaseChunker, Chunk

# Minimum characters per chunk to avoid tiny fragments
_MIN_CHUNK_CHARS = 100


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
    """

    def __init__(self, embeddings, breakpoint_threshold: float = 0.3, min_chunk_chars: int = _MIN_CHUNK_CHARS):
        self.embeddings = embeddings
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_chars = min_chunk_chars

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
                merged.append(group)

        return [
            self._make_chunk(" ".join(group), source_doc, i)
            for i, group in enumerate(merged)
        ]

    def _make_chunk(self, content: str, source_doc: dict, index: int) -> Chunk:
        return Chunk(
            content=content,
            metadata={
                "source": source_doc.get("url", ""),
                "title": source_doc.get("title", ""),
                "chunk_index": index,
            },
        )
