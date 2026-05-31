"""Base class for retrieval strategies."""

from abc import ABC, abstractmethod

from ..chunking.base import Chunk


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    @abstractmethod
    async def retrieve(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Retrieve and rank chunks relevant to the query."""
        ...
