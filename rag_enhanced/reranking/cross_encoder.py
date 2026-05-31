"""Cross-Encoder reranker — higher quality, requires sentence-transformers model."""

from __future__ import annotations

import asyncio

from ..chunking.base import Chunk
from .base import BaseReranker

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Rerank using a cross-encoder model for query-document pair scoring.

    The `model` parameter accepts either:
        - A pre-loaded cross-encoder model with a `predict(pairs)` method
        - None — will lazy-load sentence_transformers.CrossEncoder on first use
    """

    def __init__(self, model=None, top_k: int = 10, model_name: str = _DEFAULT_MODEL):
        self._model = model
        self.top_k = top_k
        self.model_name = model_name

    @property
    def model(self):
        """Lazy-load the cross-encoder model on first access."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [[query, c.content] for c in chunks]
        scores = await asyncio.to_thread(self.model.predict, pairs)

        scored: list[tuple[float, Chunk]] = []
        for chunk, score in zip(chunks, scores):
            ranked_chunk = Chunk(
                content=chunk.content,
                metadata=dict(chunk.metadata),
                vector_score=chunk.vector_score,
                bm25_score=chunk.bm25_score,
                hybrid_score=chunk.hybrid_score,
                rerank_score=float(score),
            )
            scored.append((float(score), ranked_chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[: self.top_k]]
