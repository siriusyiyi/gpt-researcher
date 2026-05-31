"""Hybrid BM25 + vector retrieval with RRF or weighted fusion."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from ..chunking.base import Chunk
from ..utils.text import cosine_similarity
from .base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Combine BM25 keyword search with vector similarity search.

    Fusion modes:
        - "rrf" (default): Reciprocal Rank Fusion — rank-based, no weight tuning.
        - "weighted": Linear combination of normalised BM25 and vector scores.
    """

    def __init__(
        self,
        embeddings,
        top_k: int = 20,
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self.embeddings = embeddings
        self.top_k = top_k
        self.fusion_mode = fusion_mode
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    async def retrieve(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        # Clone chunks to avoid mutating input
        results = [Chunk(content=c.content, metadata=dict(c.metadata)) for c in chunks]

        # BM25 search
        bm25_ranked = self._bm25_search(query, results)

        # Vector search
        vector_ranked = await self._vector_search(query, results)

        # Fusion
        if self.fusion_mode == "rrf":
            fused = self._rrf_fusion(bm25_ranked, vector_ranked, results)
        else:
            fused = self._weighted_fusion(results)

        # Sort by hybrid_score descending, take top_k
        fused.sort(key=lambda c: c.hybrid_score, reverse=True)
        return fused[: self.top_k]

    def _bm25_search(self, query: str, chunks: list[Chunk]) -> list[int]:
        """Run BM25 and return chunk indices ranked by score."""
        tokenized_corpus = [c.content.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        for i, score in enumerate(scores):
            chunks[i].bm25_score = float(score)
        ranked = sorted(range(len(chunks)), key=lambda i: chunks[i].bm25_score, reverse=True)
        return ranked

    async def _vector_search(self, query: str, chunks: list[Chunk]) -> list[int]:
        """Embed query and chunks, compute cosine similarity, return ranked indices."""
        query_emb = await self.embeddings.aembed_query(query)
        contents = [c.content for c in chunks]
        if not contents:
            return []
        chunk_embs = await self.embeddings.aembed_documents(contents)
        for i, emb in enumerate(chunk_embs):
            sim = cosine_similarity(query_emb, emb)
            chunks[i].vector_score = sim
        ranked = sorted(range(len(chunks)), key=lambda i: chunks[i].vector_score, reverse=True)
        return ranked

    def _rrf_fusion(self, bm25_ranked: list[int], vector_ranked: list[int], chunks: list[Chunk]) -> list[Chunk]:
        """Reciprocal Rank Fusion: RRF_score = 1/(k + rank)."""
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked)}
        vec_rank_map = {idx: rank for rank, idx in enumerate(vector_ranked)}
        for i in range(len(chunks)):
            rrf_score = 0.0
            if i in bm25_rank_map:
                rrf_score += 1.0 / (self.rrf_k + bm25_rank_map[i] + 1)
            if i in vec_rank_map:
                rrf_score += 1.0 / (self.rrf_k + vec_rank_map[i] + 1)
            chunks[i].hybrid_score = rrf_score
        return chunks

    def _weighted_fusion(self, chunks: list[Chunk]) -> list[Chunk]:
        """Weighted linear combination of normalised BM25 and vector scores."""
        bm25_scores = [c.bm25_score for c in chunks]
        vec_scores = [c.vector_score for c in chunks]

        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
        max_vec = max(vec_scores) if vec_scores and max(vec_scores) > 0 else 1.0

        for c in chunks:
            norm_bm25 = c.bm25_score / max_bm25
            norm_vec = c.vector_score / max_vec
            c.hybrid_score = self.bm25_weight * norm_bm25 + self.vector_weight * norm_vec
        return chunks
