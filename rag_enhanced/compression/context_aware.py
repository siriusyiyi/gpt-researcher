"""Context-aware compression — dedup, relevance filter, top-k selection."""

from __future__ import annotations

from ..chunking.base import Chunk
from ..utils.text import cosine_similarity
from .base import BaseCompressor

_FAST_PATH_THRESHOLD = 2000


class ContextAwareCompressor(BaseCompressor):
    """Compress chunks with cross-chunk deduplication and relevance-based selection.

    Pipeline:
        1. Small-doc fast path: If total content < threshold, return directly.
        2. Deduplicate: Embed all chunks, merge near-duplicates
           (similarity > dedup_threshold), keeping the one with higher rerank_score.
        3. Sort by rerank_score (fallback to hybrid_score if rerank_score is 0).
        4. Take top_k and format as context string.
    """

    def __init__(
        self,
        embeddings,
        similarity_threshold: float = 0.35,
        dedup_threshold: float = 0.85,
        max_results: int = 10,
    ):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.max_results = max_results

    async def compress(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return ""

        total_chars = sum(len(c.content) for c in chunks)
        if total_chars < _FAST_PATH_THRESHOLD and len(chunks) <= self.max_results:
            return self._format_output(chunks[: self.max_results])

        # Reuse cached embeddings where available, only embed the rest
        to_embed_indices = [i for i, c in enumerate(chunks) if c.embedding is None]
        if to_embed_indices:
            contents = [chunks[i].content for i in to_embed_indices]
            new_embs = await self.embeddings.aembed_documents(contents)
            for idx, emb in zip(to_embed_indices, new_embs):
                chunks[idx].embedding = emb

        embs = [c.embedding for c in chunks]

        deduped = self._deduplicate(chunks, embs)
        deduped.sort(key=self._sort_key, reverse=True)
        selected = deduped[: self.max_results]
        return self._format_output(selected)

    def _deduplicate(self, chunks: list[Chunk], embeddings: list[list[float]]) -> list[Chunk]:
        if len(chunks) <= 1:
            return chunks

        keep: list[bool] = [True] * len(chunks)
        for i in range(len(chunks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(chunks)):
                if not keep[j]:
                    continue
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim > self.dedup_threshold:
                    score_i = self._sort_key(chunks[i])
                    score_j = self._sort_key(chunks[j])
                    if score_i >= score_j:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [c for c, k in zip(chunks, keep) if k]

    @staticmethod
    def _sort_key(chunk: Chunk) -> float:
        if chunk.rerank_score > 0:
            return chunk.rerank_score
        if chunk.hybrid_score > 0:
            return chunk.hybrid_score
        return chunk.vector_score

    @staticmethod
    def _format_output(chunks: list[Chunk]) -> str:
        parts = []
        for c in chunks:
            source = c.metadata.get("source", "unknown")
            title = c.metadata.get("title", "")
            header = f"Source: {source}"
            if title:
                header += f"\nTitle: {title}"
            parts.append(f"{header}\nContent: {c.content}")
        return "\n".join(parts)
