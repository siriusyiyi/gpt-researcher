"""Base class for reranking strategies."""

from abc import ABC, abstractmethod

from ..chunking.base import Chunk


class BaseReranker(ABC):
    """Abstract base class for all reranking strategies."""

    @abstractmethod
    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Re-rank chunks by relevance to the query."""
        ...
