"""Embedding similarity reranker — zero extra cost, reuses existing embeddings."""

from __future__ import annotations

from ..chunking.base import Chunk
from ..utils.text import cosine_similarity
from .base import BaseReranker


class EmbeddingReranker(BaseReranker):
    """Rerank chunks by computing query-chunk embedding cosine similarity."""

    def __init__(self, embeddings, top_k: int = 10):
        self.embeddings = embeddings
        self.top_k = top_k

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        query_emb = await self.embeddings.aembed_query(query)
        contents = [c.content for c in chunks]
        chunk_embs = await self.embeddings.aembed_documents(contents)

        scored: list[tuple[float, Chunk]] = []
        for chunk, emb in zip(chunks, chunk_embs):
            sim = cosine_similarity(query_emb, emb)
            ranked_chunk = Chunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
                vector_score=chunk.vector_score,
                bm25_score=chunk.bm25_score,
                hybrid_score=chunk.hybrid_score,
                rerank_score=sim,
            )
            scored.append((sim, ranked_chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[: self.top_k]]
