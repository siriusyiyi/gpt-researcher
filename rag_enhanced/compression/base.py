"""Base class for compression strategies."""

from abc import ABC, abstractmethod

from ..chunking.base import Chunk


class BaseCompressor(ABC):
    """Abstract base class for all compression strategies."""

    @abstractmethod
    async def compress(self, query: str, chunks: list[Chunk]) -> str:
        """Compress chunks into a formatted context string."""
        ...
